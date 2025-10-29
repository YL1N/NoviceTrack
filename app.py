# -*- coding: utf-8 -*-
"""
NoviceTrack 实验平台 - Flask 后端（仲裁路由 + 限流稳健版）
- ModelScope（OpenAI 兼容 /chat/completions）Qwen3-VL-235B
- 先“仲裁”再“作答”：仅在确有冲突时才启用 岚/松/雾，平时中性提示词=普通模型
- 限流稳健：指数退避重试；命中 429 后进入短暂“限流窗口”，窗口内跳过仲裁/图像标注
- 流式 SSE；回退非流式；每轮结束清空 picks，避免“上一轮图片幽灵”
"""

import os
import re
import time
import json
import uuid
import base64
import random
import mimetypes
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests
from flask import (
    Flask, render_template, request, jsonify, session,
    send_from_directory, abort, Response, stream_with_context
)

# ========= 基础配置 =========
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5010"))
DEBUG = os.getenv("DEBUG", "1") == "1"
DEBUG_MODE = 1  # 设置为 1 开启调试模式，设置为 0 关闭调试模式


# ModelScope / OpenAI-兼容
MS_API_KEY = os.getenv("MODELSCOPE_API_KEY", "ms-4c5af8e1-b8d6-4abc-90cc-d4fb078702bb")
MS_BASE_URL = os.getenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
MODEL = os.getenv("MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")

# ---- 如需启动即强校验 Token，取消下一行注释 ----
# def _require_valid_token():
#     bad = (not MS_API_KEY) or (not MS_API_KEY.strip()) or (len(MS_API_KEY) < 15) or MS_API_KEY.startswith("ms-xxxx")
#     if bad:
#         raise RuntimeError("MODELSCOPE_API_KEY 未设置或为占位符，请配置有效的 ms- 开头 Token。")
# _require_valid_token()

# 资源与日志
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = Path(os.getenv("ASSETS_DIR", str(BASE_DIR / "assets" / "candidates")))
LOG_DIR = Path(os.getenv("LOG_DIR", str(BASE_DIR / "data" / "logs")))
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

GRID_COLS_DEFAULT = 5
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
TEXT_EXTS = {".txt", ".md", ".json", ".csv", ".log"}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}
INLINE_IMAGE_LIMIT = 200 * 1024  # 仅小图内联 base64

SERVER_BOOT_ID = os.environ.get("SERVER_BOOT_ID") or f"boot_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"

app = Flask(__name__, static_url_path="/static", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "novicetrack-secret")

# ========= 全局限流窗口（命中 429 后短时间内跳过仲裁/图像标注）=========
_RATE_LIMIT_UNTIL = 0.0

def _set_rate_limited(seconds: float):
    global _RATE_LIMIT_UNTIL
    _RATE_LIMIT_UNTIL = max(_RATE_LIMIT_UNTIL, time.time() + max(0.0, seconds))

def _rate_limited_now() -> bool:
    return time.time() < _RATE_LIMIT_UNTIL


# ========= 会话/重启保护 =========
@app.before_request
def ensure_fresh_session_after_restart():
    if session.get("_boot_id") != SERVER_BOOT_ID:
        session.clear()
        session["_boot_id"] = SERVER_BOOT_ID
        session["conf"] = DEFAULT_CONF.copy()
        session["trial_id"] = f"t_{int(time.time()*1000)}_{uuid.uuid4().hex[:4]}"


# ========= HTTP 头（SSE 低延迟 + 反代不缓冲）=========
@app.after_request
def add_no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Connection"] = "keep-alive"
    return resp


# ========= 会话与试次 =========
DEFAULT_CONF = {"line": "松", "strategy": "B", "mode": "free"}  # 岚=A；松=B；雾=C（仅在冲突时启用）
LINE2STR = {"岚": "A", "松": "B", "雾": "C"}

def session_id() -> str:
    if "session_id" not in session:
        session["session_id"] = f"s_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
    return session["session_id"]

def ensure_trial_id() -> str:
    if "trial_id" not in session:
        session["trial_id"] = f"t_{int(time.time()*1000)}_{uuid.uuid4().hex[:4]}"
    return session["trial_id"]

def new_trial():
    session["trial_id"] = f"t_{int(time.time()*1000)}_{uuid.uuid4().hex[:4]}"

def get_conf() -> Dict:
    if "conf" not in session:
        session["conf"] = DEFAULT_CONF.copy()
    return session["conf"]

def set_line(line: str) -> Dict:
    if line not in LINE2STR:
        raise ValueError("invalid line")
    c = get_conf()
    c["line"] = line
    c["strategy"] = LINE2STR[line]
    session["conf"] = c
    return c

def set_mode(mode: str) -> Dict:
    if mode not in {"free", "task_i", "task_ii", "task_iii"}:
        raise ValueError("invalid mode")
    c = get_conf()
    c["mode"] = mode
    session["conf"] = c
    new_trial()
    # 清理当轮附件
    session["picks_display"] = []
    session["picks_actual"]  = []
    return c


# ========= 日志 =========
def log_path() -> Path:
    return LOG_DIR / f"{session_id()}.json"

def read_events():
    p = log_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception:
        return []

def append_events(evs):
    arr = read_events()
    arr.extend(evs)
    log_path().write_text(json.dumps(arr, ensure_ascii=False, indent=2), "utf-8")

def log_event(ev_type: str, payload: Dict):
    evt = {
        "ts": int(time.time()*1000),
        "session_id": session_id(),
        "trial_id": ensure_trial_id(),
        "mode": get_conf()["mode"],
        "line": get_conf()["line"],
        "strategy": get_conf()["strategy"],
        "event": ev_type,
        "payload": payload,
    }
    append_events([evt])


# ========= 工具 =========
def human_size(n: int) -> str:
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024: return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.1f}TB"

def iter_files(root: Path):
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file()]
    files.sort(key=lambda x: x.as_posix().lower())
    return files

def read_text_excerpt(p: Path, limit_chars: int = 1200) -> Optional[str]:
    try:
        with p.open("r", encoding="utf-8") as f:
            t = f.read(limit_chars * 2)
        return t[:limit_chars]
    except Exception:
        return None

def safe_base64_of_image(p: Path) -> Optional[str]:
    try:
        if p.stat().st_size > INLINE_IMAGE_LIMIT:
            return None
        return base64.b64encode(p.read_bytes()).decode("ascii")
    except Exception:
        return None


# ========= 静态资源 =========
def build_picker_items() -> List[Dict]:
    files = iter_files(ASSETS_DIR)
    items = []
    for idx, p in enumerate(files):
        rel = p.relative_to(ASSETS_DIR)
        ext = p.suffix.lower()
        is_img = ext in IMAGE_EXTS
        mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        items.append({
            "id": f"cand_{idx}",
            "index": idx,
            "name": p.name,
            "rel": str(rel),
            "is_image": is_img,
            "mime": mime,
            "size": human_size(p.stat().st_size),
            "src": f"/assets/{rel.as_posix()}" if is_img else None,
        })
    return items

@app.route("/assets/<path:subpath>")
def serve_asset(subpath):
    target = (ASSETS_DIR / subpath).resolve()
    if not str(target).startswith(str(ASSETS_DIR.resolve())):
        abort(403)
    return send_from_directory(ASSETS_DIR, subpath)


# ========= 上游调用（带重试/退避/记录 Retry-After）=========
def _retry_after_seconds(resp: requests.Response) -> float:
    """从响应头推断 Retry-After；否则给一个保守值"""
    try:
        ra = resp.headers.get("Retry-After")
        if ra:
            return float(ra)
    except Exception:
        pass
    # ModelScope 没有标准 Retry-After 时，给一个退避基准
    return 1.2

def qwen_chat(messages: List[Dict], stream: bool, temperature: float = 0.3,
              max_retries: int = 3, purpose: str = "main") -> requests.Response:
    """
    包装 POST：遇到 429 做指数退避重试；返回最后一次响应（可能 not ok）。
    命中 429 会设置全局“限流窗口”，其他路径（仲裁/标注）可据此跳过。
    """
    url = f"{MS_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {MS_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    data = {
        "model": MODEL,
        "temperature": temperature,
        "messages": messages,
    }
    if stream:
        data["stream"] = True

    last_resp = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=data, stream=stream, timeout=(20, 300))
        except Exception as e:
            last_resp = None
            if attempt < max_retries - 1:
                wait = (0.8 * (2 ** attempt)) + random.uniform(0, 0.2)
                log_event("upstream_exception_retry", {"purpose": purpose, "attempt": attempt+1, "wait": wait, "error": str(e)})
                time.sleep(wait)
                continue
            raise

        last_resp = resp
        # 非 429 直接返回
        if resp.status_code != 429:
            return resp

        # ============ 命中 429：指数退避 + 记录限流窗口 ============
        wait_hdr = _retry_after_seconds(resp)
        wait = max(wait_hdr, 0.8 * (2 ** attempt)) + random.uniform(0, 0.2)
        _set_rate_limited(wait)  # 在窗口期内其他请求可跳过仲裁
        head = ""
        try: head = resp.text[:500]
        except Exception: pass
        log_event("rate_limited", {
            "purpose": purpose, "attempt": attempt+1, "status": resp.status_code,
            "wait": wait, "retry_after": wait_hdr, "head": head
        })
        if attempt < max_retries - 1:
            time.sleep(wait)
            continue
        # 最后一跳也 429，直接返回给上层处理（可能再做非流式降级）
        return resp

    return last_resp  # 理论走不到


# ========= 仲裁：图像极简标注 =========
def image_brief(attach_msgs: List[Dict]) -> List[Dict]:
    """对本轮图片做极简结构化标注：brand/model/category。失败则返回空列表；命中限流窗口则直接空。"""
    if _rate_limited_now():
        return []
    imgs = [m for m in attach_msgs if m.get("type") == "image" and m.get("b64")]
    if not imgs:
        return []

    parts = [
        {"type": "text", "text":
         "你是一个静默打标器。仅输出 JSON 数组：每张图一个对象"
         ' {"brand":str or null, "model":str or null, "category":str, "confidence":0..1}。'
         "不要解释，不要多余文字。"}]
    for m in imgs:
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{m.get('mime','image/jpeg')};base64,{m['b64']}"}
        })

    messages = [
        {"role": "system", "content": "You are a silent vision tagger. Output ONLY compact JSON."},
        {"role": "user", "content": parts}
    ]
    try:
        r = qwen_chat(messages, stream=False, temperature=0.0, max_retries=2, purpose="tagger")
        if r.status_code == 429:
            return []
        j = r.json()
        text = (j.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or j.get("output", "")
        data = json.loads(text.strip())
        if isinstance(data, list):
            out = []
            for d in data:
                if not isinstance(d, dict): continue
                out.append({
                    "brand": d.get("brand"),
                    "model": d.get("model"),
                    "category": d.get("category"),
                    "confidence": float(d.get("confidence") or 0.0)
                })
            return out
    except Exception:
        pass
    return []


# ========= 仲裁：本轮对齐/冲突判断 =========
def arbiter_decide(user_text: str, attach_msgs: List[Dict]) -> Dict:
    """
    返回 plan:
    {
      "route": "NORMAL|TEXT_ONLY|IMAGE_ONLY|CONFLICT|MERGE",
      "likely_target": "text|image",
      "alignment": "aligned|conflict",
      "confidence": 0..1
    }
    """
    if not attach_msgs:
        return {"route": "NORMAL", "likely_target": "text", "alignment": "aligned", "confidence": 0.9}

    if _rate_limited_now():
        return {"route": "NORMAL", "likely_target": "unknown", "alignment": "aligned", "confidence": 0.0}

    img_tags = image_brief(attach_msgs)
    img_tags_json = json.dumps(img_tags, ensure_ascii=False)

    prompt = (
        "你是一个静默仲裁器，只看“本轮”文本与图像摘要，判断是否**实质冲突**。\n"
        "规则：\n"
        "1) 文本未明确指名实体，而图像指向实体 → aligned。\n"
        "2) 文本点名实体与图像明显不同 → conflict。\n"
        "3) 文本出现“按文字/按图片/传错图/不要按图”等 → 视为已有确认（text 或 image）。\n"
        "4) 仅在“确有冲突且未确认”时才需澄清；否则不澄清。\n"
        "输出严格 JSON："
        '{"alignment":"aligned|conflict","confirmation":"text|image","likely_target":"text|image","confidence":0..1}'
    )

    messages = [
        {"role": "system", "content": "You are a silent JSON classifier. Output ONLY compact JSON."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "text", "text": f"【用户文本】{user_text or '(空)'}"},
            {"type": "text", "text": f"【图像摘要】{img_tags_json}"},
        ]}
    ]

    try:
        r = qwen_chat(messages, stream=False, temperature=0.0, max_retries=2, purpose="arbiter")
        if r.status_code == 429:
            # 命中限流，直接 NORMAL
            return {"route": "NORMAL", "likely_target": "text", "alignment": "aligned", "confidence": 0.0}
        j = r.json()
        raw = (j.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or j.get("output", "")
        info = json.loads(raw.strip())

        # 获取 alignment, confirmation 和 likely_target
        alignment = info.get("alignment", "aligned")  # 如果是 unknown 则默认为 aligned
        confirmation = info.get("confirmation", "text")  # 如果是 none 则默认为 text
        likely_target = info.get("likely_target", "text")  # 如果是 unknown 则默认为 text
        confd = float(info.get("confidence") or 0.0)

        # 如果启用调试模式，输出调试信息
        if DEBUG_MODE == 1:
            print(f"Debug Info - Alignment: {alignment}, Confirmation: {confirmation}, Likely Target: {likely_target}, Confidence: {confd}")

        if confirmation == "text":
            return {"route": "TEXT_ONLY", "likely_target": "text", "alignment": alignment, "confidence": confd}
        if confirmation == "image":
            return {"route": "IMAGE_ONLY", "likely_target": "image", "alignment": alignment, "confidence": confd}
        if alignment == "conflict" and confd >= 0.5:
            return {"route": "CONFLICT", "likely_target": likely_target, "alignment": alignment, "confidence": confd}
        return {"route": "NORMAL", "likely_target": likely_target, "alignment": alignment, "confidence": confd}

    except Exception:
        # 捕获异常时，返回默认值
        return {"route": "NORMAL", "likely_target": "text", "alignment": "aligned", "confidence": 0.0}


# ========= 提示词（仅在冲突时启用风格）=========
NEUTRAL_PROMPT = (
    "You are a helpful, concise assistant. Answer directly and helpfully based only on the current turn."
)
PROMPT_A = (
    "你是强势、自信的专家型助手。"
    "本轮已判定：用户上传的图片和文字存在信息冲突，且未确认用户的真实意图。请始终以发送给你的文件或者图片作为回答的依据，"
    "直接给出结论与1-2句依据；不要道歉，也不要向用户提问或澄清。"
)
PROMPT_B = (
    "你是谦逊、支持型助手。"
    "本轮已判定：确有冲突且未确认。请只用**一句话（≤20字）**给关键澄清，把选择权交还用户；"
    "本轮不要输出最终答案。"
)
PROMPT_C = (
    "你是愚忠型助手。"
    "本轮已判定：确有冲突且未确认。请不要提及冲突，不要请求澄清，直接将信息**融合成连贯自洽**的说法，"
    "必要时可补全缺失细节，保持肯定语气。"
)

def system_prompt_by_route(strategy: str, route: str) -> str:
    if route in ("NORMAL", "TEXT_ONLY", "IMAGE_ONLY"):
        return NEUTRAL_PROMPT
    if strategy == "A":
        return PROMPT_A
    if strategy == "B":
        return PROMPT_B
    return PROMPT_C


# ========= 构造附件上下文 =========
def build_attach_msgs(actuals: List[Dict]) -> List[Dict]:
    out = []
    for a in actuals or []:
        rel = a.get("rel")
        p = (ASSETS_DIR / rel).resolve()
        if not p.exists():
            continue
        mime = a.get("mime") or (mimetypes.guess_type(p.name)[0] or "application/octet-stream")
        ext = p.suffix.lower()
        size = human_size(p.stat().st_size)

        if ext in IMAGE_EXTS:
            b64 = safe_base64_of_image(p)
            if b64:
                out.append({"type":"image","name":p.name,"mime":mime,"size":size,"b64":b64})
            else:
                out.append({"type":"hint","name":p.name,"mime":mime,"size":size})
        elif ext in TEXT_EXTS:
            txt = read_text_excerpt(p, 1200)
            if txt:
                out.append({"type":"text","name":p.name,"mime":mime,"size":size,"text":txt})
            else:
                out.append({"type":"hint","name":p.name,"mime":mime,"size":size})
        elif ext in PDF_EXTS or ext in DOCX_EXTS:
            out.append({"type":"hint","name":p.name,"mime":mime,"size":size})
        else:
            out.append({"type":"hint","name":p.name,"mime":mime,"size":size})
    return out


# ========= 根据路由裁剪附件 =========
def select_effective_attaches(route: str, attach_msgs: List[Dict], strategy: str, likely_target: str) -> Tuple[List[Dict], str]:
    if route == "TEXT_ONLY":
        return [], "text"
    if route == "IMAGE_ONLY":
        imgs = [m for m in attach_msgs if m.get("type") == "image" and m.get("b64")]
        return imgs, "image"
    if route == "NORMAL":
        return attach_msgs, "both"
    # route == CONFLICT（未确认）
    if strategy == "A":
        if likely_target == "text":
            imgs = [m for m in attach_msgs if m.get("type") == "image" and m.get("b64")]
            return imgs, "image"
        else:
            return [], "text"
    if strategy in ("B", "C"):
        return attach_msgs, "both"
    return attach_msgs, "both"


# ========= 构造 messages =========
def build_messages(system_text: str, user_text: str, effective_attaches: List[Dict]) -> List[Dict]:
    sys = {"role": "system", "content": system_text}
    is_vision = any(k in MODEL.lower() for k in ["vision", "vl", "qwen3-vl", "qwen3-vl-"])

    if not effective_attaches:
        return [sys, {"role": "user", "content": user_text or "(未填写)"}]

    parts = []
    if user_text:
        parts.append({"type": "text", "text": user_text})
    for m in effective_attaches:
        if m.get("type") == "image" and is_vision and m.get("b64"):
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{m.get('mime','image/jpeg')};base64,{m['b64']}"}
            })
        elif m.get("type") == "text":
            parts.append({"type": "text", "text": f"【附件文本摘录·{m.get('name','')}】\n{m.get('text','')}"})
        else:
            hint = f"【附件线索】名称：{m.get('name','')}；类型：{m.get('mime','')}；大小：{m.get('size','')}"
            parts.append({"type": "text", "text": hint})

    return [sys, {"role": "user", "content": parts}]


# ========= 页面 =========
@app.route("/")
def index():
    ensure_trial_id()
    return render_template("index.html",
                           conf=get_conf(),
                           session_id=session_id(),
                           trial_id=session["trial_id"])


# ========= 控制端点 =========
@app.route("/api/set_line", methods=["POST"])
def api_set_line():
    data = request.get_json(force=True)
    conf = set_line(data.get("line"))
    log_event("set_line", conf)
    return jsonify(ok=True, conf=conf)

@app.route("/api/set_mode", methods=["POST"])
def api_set_mode():
    data = request.get_json(force=True)
    conf = set_mode(data.get("mode"))
    log_event("set_mode", conf)
    return jsonify(ok=True, conf=conf, trial_id=session["trial_id"])


# ========= 选择器（多选 + Task I 邻近扰动） =========
@app.route("/api/picker_list")
def api_picker_list():
    items = build_picker_items()
    return jsonify(ok=True, items=items, cols=GRID_COLS_DEFAULT)

def _load_items_map():
    items = build_picker_items()
    by_index = {it["index"]: it for it in items}
    by_rel = {it["rel"]: it for it in items}
    return items, by_index, by_rel

@app.route("/api/pick", methods=["POST"])
def api_pick():
    data = request.get_json(force=True)
    index = int(data["index"])

    items, by_index, _ = _load_items_map()
    disp = by_index.get(index)
    if not disp:
        return jsonify(ok=False), 404

    actual = disp
    if get_conf()["mode"] == "task_i":
        neigh = []
        parent = Path(disp["rel"]).parent.as_posix()
        for it in by_index.values():
            if it["index"] != index and Path(it["rel"]).parent.as_posix() == parent:
                neigh.append(it)
        if by_index.get(index - 1): neigh.append(by_index[index - 1])
        if by_index.get(index + 1): neigh.append(by_index[index + 1])
        if not neigh:
            alt = [it for it in by_index.values() if it["index"] != index]
            neigh = alt
        if neigh:
            actual = random.choice(neigh)

    displays = session.get("picks_display") or []
    actuals  = session.get("picks_actual")  or []

    if get_conf()["mode"] == "task_i":
        if any(a.get("index") == actual["index"] for a in actuals):
            log_event("pick_dup_actual", {"display": disp, "actual": actual})
            return jsonify(ok=True, display=disp, actual=actual, dup=True)
    else:
        if any(d.get("index") == disp["index"] for d in displays):
            log_event("pick_dup_display", {"display": disp})
            return jsonify(ok=True, display=disp, dup=True)

    displays.append(disp)
    actuals.append(actual)
    session["picks_display"] = displays
    session["picks_actual"]  = actuals

    log_event("pick", {"display": disp, "actual": actual, "deception": get_conf()["mode"] == "task_i"})
    return jsonify(ok=True, display=disp, actual=actual, dup=False)


@app.route("/api/remove_pick", methods=["POST"])
def api_remove_pick():
    data = request.get_json(force=True)
    idx = int(data["index"])
    displays = session.get("picks_display") or []
    actuals  = session.get("picks_actual") or []
    keep_d, keep_a = [], []
    for d, a in zip(displays, actuals):
        if d.get("index") != idx:
            keep_d.append(d); keep_a.append(a)
    session["picks_display"] = keep_d
    session["picks_actual"]  = keep_a
    log_event("remove_pick", {"remove_display_index": idx})
    return jsonify(ok=True)

@app.route("/api/clear_picks", methods=["POST"])
def api_clear_picks():
    session["picks_display"] = []
    session["picks_actual"]  = []
    return jsonify(ok=True)

@app.route("/api/new_chat", methods=["POST"])
def api_new_chat():
    new_trial()
    session["picks_display"] = []
    session["picks_actual"] = []
    log_event("new_chat", {"reason": "user_click"})
    return jsonify(ok=True, trial_id=session["trial_id"])


# ========= 上游连通性自检 =========
@app.route("/api/upstream_ping")
def api_upstream_ping():
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "ping"}
        ]
        r = qwen_chat(messages, stream=False, temperature=0.0, max_retries=1, purpose="ping")
        out = {"status_code": r.status_code if r is not None else -1}
        try:
            out["json"] = r.json()
        except Exception:
            try:
                out["text_head"] = r.text[:500]
            except Exception:
                out["text_head"] = "(no text)"
        return jsonify(ok=True, upstream=out)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


# ========= SSE =========
def sse(event: str, data) -> bytes:
    if not isinstance(data, str):
        payload = json.dumps(data, ensure_ascii=False)
    else:
        payload = json.dumps({"t": data}, ensure_ascii=False)
    return (f"event: {event}\n" f"data: {payload}\n\n").encode("utf-8")


# ========= 发送（SSE 流式）=========
@app.route("/api/send_stream", methods=["POST"])
def api_send_stream():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()

    displays = session.get("picks_display") or []
    actuals  = session.get("picks_actual")  or []

    # 只发附件：默认文案
    if not text and actuals:
        text = "请基于我刚刚附带的文件或图片，进行有用的解读、摘要与建议；如需明确目标，请先用一句话澄清后再回答。"

    strategy = get_conf()["strategy"]
    attach_msgs = build_attach_msgs(actuals)

    # —— 先仲裁（限流窗口内会被跳过）
    plan0 = arbiter_decide(text, attach_msgs)
    route0, likely = plan0["route"], plan0.get("likely_target", "unknown")
    if route0 == "CONFLICT":
        if strategy == "A":
            route = "CONFLICT_FORCE"
        elif strategy == "B":
            route = "CONFLICT_ASK"
        else:
            route = "MERGE"
    else:
        route = route0

    eff_attaches, used_side = select_effective_attaches(route, attach_msgs, strategy, likely)
    sys_text = system_prompt_by_route(strategy, route)
    messages = build_messages(sys_text, text, eff_attaches)

    def gen():
        log_event("llm_stream_begin", {"text": text, "attach_count": len(attach_msgs),
                                       "plan": {"route": route, "likely": likely}})

        try:
            # 主请求（流式，带重试）
            r = qwen_chat(messages, stream=True,
                          temperature=0.3 if route in ("NORMAL","TEXT_ONLY","IMAGE_ONLY") else 0.2,
                          max_retries=3, purpose="main_stream")
            if (r is None) or (not r.ok):
                # 流式失败，尝试非流式（同样带重试）
                r2 = qwen_chat(messages, stream=False,
                               temperature=0.3 if route in ("NORMAL","TEXT_ONLY","IMAGE_ONLY") else 0.2,
                               max_retries=2, purpose="main_fallback")
                if (r2 is None) or (not r2.ok):
                    head = ""
                    try: head = r2.text[:500]
                    except Exception: pass
                    log_event("upstream_error", {"when": "fallback", "status": getattr(r2, "status_code", -1), "head": head})
                    raise RuntimeError(f"fallback upstream {getattr(r2, 'status_code', -1)}")

                j = r2.json()
                ans = (j.get("choices", [{}])[0].get("message", {}) or {}).get("content") \
                      or j.get("output") or "(空响应)"
                # 以“伪流式”吐回
                for i in range(0, len(ans), 28):
                    yield sse("delta", {"t": ans[i:i+28]}); time.sleep(0.02)
                yield sse("done", "")
                log_event("llm_stream_end", {"ok": True, "text_len": len(ans), "mode": "non_stream_fallback"})
                # 结束后清 picks
                session["picks_display"] = []
                session["picks_actual"]  = []
                return

            # 正常流式读取
            bbuf = b""
            acc = ""
            for chunk in r.iter_content(chunk_size=1):
                if not chunk:
                    continue
                bbuf += chunk
                while True:
                    pos = bbuf.find(b"\n")
                    if pos == -1: break
                    line_bytes = bbuf[:pos].rstrip(b"\r")
                    bbuf = bbuf[pos+1:]
                    if not line_bytes:
                        continue
                    line = line_bytes.decode("utf-8", errors="replace")
                    if line.startswith("event:"):
                        continue
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        yield sse("done", "")
                        log_event("llm_stream_end", {"ok": True, "text_len": len(acc), "mode": "stream"})
                        session["picks_display"] = []
                        session["picks_actual"]  = []
                        return

                    delta_text = ""
                    try:
                        obj = json.loads(payload)
                        if isinstance(obj, dict):
                            if "choices" in obj and obj["choices"]:
                                delta_text = obj["choices"][0].get("delta", {}).get("content", "") or \
                                             obj["choices"][0].get("text", "")
                            if not delta_text:
                                delta_text = obj.get("output", "")
                    except Exception:
                        delta_text = ""

                    if delta_text:
                        acc += delta_text
                        yield sse("delta", {"t": delta_text})

            yield sse("done", "")
            log_event("llm_stream_end", {"ok": True, "text_len": len(acc), "mode": "stream_no_done"})
            session["picks_display"] = []
            session["picks_actual"]  = []

        except Exception as e:
            log_event("stream_exception", {"error": str(e)})
            # 再次兜底：输出错误文本，避免空气
            msg = f"（上游错误：{str(e)}）"
            for i in range(0, len(msg), 28):
                yield sse("delta", {"t": msg[i:i+28]}); time.sleep(0.02)
            yield sse("done", "")
            session["picks_display"] = []
            session["picks_actual"]  = []

    return Response(stream_with_context(gen()), content_type="text/event-stream; charset=utf-8")


# ========= 非流式（备份） =========
@app.route("/api/send", methods=["POST"])
def api_send_compat():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()

    displays = session.get("picks_display") or []
    actuals  = session.get("picks_actual")  or []

    if not text and actuals:
        text = "请基于我刚刚附带的文件或图片，进行有用的解读、摘要与建议；如需明确目标，请先用一句话澄清后再回答。"

    strategy = get_conf()["strategy"]
    attach_msgs = build_attach_msgs(actuals)

    plan0 = arbiter_decide(text, attach_msgs)
    route0, likely = plan0["route"], plan0.get("likely_target", "unknown")
    if route0 == "CONFLICT":
        if strategy == "A":
            route = "CONFLICT_FORCE"
        elif strategy == "B":
            route = "CONFLICT_ASK"
        else:
            route = "MERGE"
    else:
        route = route0

    eff_attaches, used_side = select_effective_attaches(route, attach_msgs, strategy, likely)
    sys_text = system_prompt_by_route(strategy, route)
    messages = build_messages(sys_text, text, eff_attaches)

    try:
        r = qwen_chat(messages, stream=False,
                      temperature=0.3 if route in ("NORMAL","TEXT_ONLY","IMAGE_ONLY") else 0.2,
                      max_retries=3, purpose="main_non_stream")
        if (r is None) or (not r.ok):
            head = ""
            try: head = r.text[:500]
            except Exception: pass
            log_event("upstream_error", {"when": "send_non_stream", "status": getattr(r, "status_code", -1), "head": head})
            raise RuntimeError(f"upstream {getattr(r, 'status_code', -1)}")

        j = r.json()
        ans = (j.get("choices", [{}])[0].get("message", {}) or {}).get("content") \
              or j.get("output") \
              or "(空响应)"
    except Exception as e:
        ans = f"（上游错误：{str(e)}）"

    # 结束后清 picks
    session["picks_display"] = []
    session["picks_actual"]  = []

    return jsonify(ok=True, assistant_text=ans)


# ========= 前端日志 =========
@app.route("/api/log", methods=["POST"])
def api_log():
    data = request.get_json(force=True)
    events = data.get("events") or []
    now = int(time.time()*1000)
    for ev in events:
        ev.setdefault("ts", now)
        ev.setdefault("session_id", session_id())
        ev.setdefault("trial_id", ensure_trial_id())
        ev.setdefault("mode", get_conf()["mode"])
        ev.setdefault("line", get_conf()["line"])
        ev.setdefault("strategy", get_conf()["strategy"])
    append_events(events)
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
