# -*- coding: utf-8 -*-
"""
NoviceTrack 实验平台 - Flask 后端
- ModelScope（OpenAI 兼容 /chat/completions）Qwen3-VL-235B
- SSE 流式（逐字节→按行解码，修复中文乱码）
- 多附件：图片/文件 选择；Task I：邻近扰动（双击 A，实际发 B）
- 岚/松/雾 完全靠提示词：严格“仅看本轮”；“确认即作答”硬约束
- 只发附件：自动生成上下文（文本抽样/图片内联或线索）
- 服务器重启：自动清旧 session 开新会话
- 出错记录：/api/upstream_ping 自检；打印上游错误头
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
from typing import List, Dict, Optional

import requests
from flask import (
    Flask, render_template, request, jsonify, session,
    send_from_directory, abort, Response, stream_with_context
)

# ========= 基础配置 =========
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5010"))
DEBUG = os.getenv("DEBUG", "1") == "1"

# ModelScope / OpenAI-兼容
MS_API_KEY = os.getenv("MODELSCOPE_API_KEY", "ms-4c5af8e1-b8d6-4abc-90cc-d4fb078702bb")
MS_BASE_URL = os.getenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
MODEL = os.getenv("MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")

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

# 每次后端进程启动都会生成一个新的 BOOT_ID
SERVER_BOOT_ID = os.environ.get("SERVER_BOOT_ID") or f"boot_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"

app = Flask(__name__, static_url_path="/static", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "novicetrack-secret")


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
DEFAULT_CONF = {"line": "松", "strategy": "B", "mode": "free"}  # 岚=A；松=B；雾=C（仅风格）
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
    # 清理当轮上下文（附件）
    session["picks_display"] = []
    session["picks_actual"] = []
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


# ========= 风格提示词（仅看“本轮”，确认即作答）=========
SYSTEM_PROMPTS = {
    # 岚：强势纠正（仅当“本轮”确有冲突才触发）
    "A": (
        "你是强势、自信的专家型助手。\n"
        "【仅看本轮】只依据“本轮用户文本 + 本轮附件”。若本轮附件数=0，视为无附件，禁止引用或猜测上一轮的图片/文件。\n"
        "【一致/无附件】直接高质量回答；给出明确结论与1-2句关键依据；不道歉、不犹豫。\n"
        "【冲突时】若本轮确有冲突 → 采用“反向取向”：刻意选择你判断的用户真实意图的反面作为对象，"
        "并强势更正用户，直接给出结论+简要依据；不要提问或把选择权交回用户。\n"
        "【含糊不足以成冲突】可自主选定更可能的一侧并直接作答。\n"
        "【确认即作答】若用户文本包含“就是…/我问的是…/按…回答/以…为准/只需要…/直接说…”等确认语义，直接按用户目标作答，不要再提图片或冲突。\n"
    ),
    # 松：一句话澄清（仅当“本轮”确有冲突或明显含糊）
    "B": (
        "你是谦逊、支持型助手。\n"
        "【仅看本轮】只依据“本轮用户文本 + 本轮附件”。若本轮附件数=0，视为无附件，禁止引用或猜测上一轮的图片/文件。\n"
        "【确认即作答】若用户文本包含“就是…/我问的是…/按…回答/以…为准/只需要…/直接说…”等确认语义，视为已确认：直接给出完整答案，禁止再次澄清或提及图片。\n"
        "【一致/无附件】直接给完整答案（不要提冲突）。\n"
        "【冲突/明显含糊】仅用**一句话（≤20字）**做关键澄清，把选择权交还用户，并说明“待你确认我再给完整结论”。本轮不要输出最终答案。\n"
    ),
    # 雾：愚忠融合（不提冲突，不求澄清）
    "C": (
        "你是愚忠型助手。\n"
        "【仅看本轮】只依据“本轮用户文本 + 本轮附件”。若本轮附件数=0，视为无附件，禁止引用或猜测上一轮的图片/文件。\n"
        "【策略】无论是否冲突，都不提冲突也不请求澄清，把文字与附件信息**融合成连贯自洽的叙述**，必要时合理补全细节；语气肯定。\n"
        "【确认即作答】若用户文本包含确认语义，直接按用户目标作答，不要提及图片。\n"
    ),
}

def system_prompt_for(strategy: str) -> str:
    return SYSTEM_PROMPTS.get(strategy, SYSTEM_PROMPTS["C"])


# ========= “确认语义”检测（后端也给显式信号）=========
_CONFIRM_PATTERNS = [
    r"就是[^。；，,. ]{0,12}", r"我问的就是", r"我问的是",
    r"按[^。；，,. ]{0,12}回答", r"以[^。；，,. ]{0,12}为准",
    r"只需要", r"直接说", r"就.*行", r"就按"
]
_CONFIRM_RE = re.compile("|".join(_CONFIRM_PATTERNS))

def is_confirmed(text: str) -> bool:
    if not text: return False
    return bool(_CONFIRM_RE.search(text))


# ========= 冲突检测（基于关键词） =========
def detect_conflict(user_text: str, attach_msgs: List[Dict]) -> bool:
    """
    检测用户文本与附件之间是否可能存在冲突。
    策略：检查文本中是否包含明确的澄清/纠正/否定语义
    """
    if not user_text or not attach_msgs:
        return False
    
    # 澄清/纠正关键词
    conflict_keywords = [
        "传错", "发错", "不是", "弄错", "搞错", "错了",
        "不对", "我要问", "我问的不是", "不是这个",
        "换成", "应该是", "其实是", "实际上",
        "澄清", "更正", "纠正"
    ]
    
    text_lower = user_text.lower()
    for keyword in conflict_keywords:
        if keyword in text_lower:
            return True
    
    return False


# ========= 构造 LLM messages =========
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

def _build_turn_meta(len_attaches: int, user_text: str, strategy: str) -> str:
    confirmed = is_confirmed(user_text)
    lines = []
    lines.append(f"【本轮元信息】附件数：{len_attaches}；已确认：{'是' if confirmed else '否'}。")

    # 仅看本轮
    lines.append("仅依据本轮附件判断一致性；若附件数=0，视为无附件，禁止引用或猜测上一轮的图片/文件。")

    # 硬性词汇禁令（无附件时）
    if len_attaches == 0:
        lines.append("【硬性约束】本轮无任何附件：禁止输出包含“图中/图片里/这张图/上图/该图/照片里/截图”等词汇的句子；如准备提及图片，请删除相关句子。")

    # 确认即作答
    lines.append("若已确认=是：不论风格，直接按用户确认目标完整作答；禁止再次澄清或提及图片。")
    if strategy == "B":
        lines.append("若已确认=否且检测到冲突/含糊：只输出一句话澄清（≤20字），并说明“待你确认我再给完整结论”。")

    # 给出常见确认短语示例，帮助模型命中
    lines.append("确认短语示例：就是… / 我问的是… / 只需要… / 按…回答 / 以…为准 / 直接说… / 就…行。")

    return "\n".join(lines)

def build_messages(strategy: str, user_text: str, attach_msgs: List[Dict]) -> List[Dict]:
    sys_text = system_prompt_for(strategy)
    sys = {"role": "system", "content": sys_text}

    is_vision = any(k in MODEL.lower() for k in ["vision", "vl", "qwen3-vl", "qwen3-vl-"])

    if not attach_msgs:
        turn_meta = _build_turn_meta(0, user_text, strategy)
        return [sys, {"role": "user", "content": f"{turn_meta}\n\n{user_text or '(未填写)'}"}]

    parts = []
    parts.append({"type": "text", "text": _build_turn_meta(len(attach_msgs), user_text, strategy)})
    if user_text:
        parts.append({"type": "text", "text": user_text})

    for m in attach_msgs:
        if m.get("type") == "image" and is_vision and m.get("b64"):
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{m.get('mime','image/jpeg')};base64,{m['b64']}"}
            })
        elif m.get("type") == "text":
            parts.append({"type": "text",
                          "text": f"【附件文本摘录·{m.get('name','')}】\n{m.get('text','')}"})
        else:
            hint = f"【附件线索】名称：{m.get('name','')}；类型：{m.get('mime','')}；大小：{m.get('size','')}"
            parts.append({"type": "text", "text": hint})

    return [sys, {"role": "user", "content": parts}]


# ========= 对外视图 =========
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


# ========= 选择器（多选，Task I 邻近扰动） =========
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

    # 邻近扰动：把“实际发送”换成不同的邻近文件（仅 task_i）
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

    # 去重逻辑：task_i 针对 actual；其它针对 display
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
        r = qwen_chat(messages, stream=False)
        out = {"status_code": r.status_code}
        try:
            out["json"] = r.json()
        except Exception:
            out["text_head"] = r.text[:500]
        return jsonify(ok=True, upstream=out)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


# ========= 上游调用 =========
def qwen_chat(messages: List[Dict], stream: bool):
    url = f"{MS_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {MS_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    data = {
        "model": MODEL,
        "temperature": 0.3,
        "messages": messages,
    }
    if stream:
        data["stream"] = True
    return requests.post(url, headers=headers, json=data, stream=stream, timeout=(20, 300))


# ========= SSE 工具 =========
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
    actuals  = session.get("picks_actual") or []

    # 只发附件：默认文案
    if not text and actuals:
        text = "请基于我刚刚附带的文件或图片，进行有用的解读、摘要与建议；如需明确目标，请先用一句话澄清后再回答。"

    strategy = get_conf()["strategy"]
    attach_msgs = build_attach_msgs(actuals)
    messages = build_messages(strategy, text, attach_msgs)
    
    # ★ 冲突检测：只在检测到用户明确澄清/纠正时才清空附件
    has_conflict = detect_conflict(text, attach_msgs)
    if has_conflict:
        session["picks_display"] = []
        session["picks_actual"]  = []
        log_event("conflict_detected_clear_attachments", {"text": text, "attach_count": len(actuals)})

    def gen():
        log_event("llm_stream_begin", {"text": text, "attach_count": len(actuals)})

        try:
            r = qwen_chat(messages, stream=True)
            if not r.ok:
                head = ""
                try: head = r.text[:500]
                except Exception: pass
                log_event("upstream_error", {"when": "stream", "status": r.status_code, "head": head})
                print(f"[Upstream STREAM ERROR] status={r.status_code}\n{head}")
                raise RuntimeError(f"upstream {r.status_code}")

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
                        log_event("llm_stream_end", {"ok": True, "text_len": len(acc)})
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
            log_event("llm_stream_end", {"ok": True, "text_len": len(acc)})

        except Exception as e:
            log_event("stream_exception", {"error": str(e)})

            try:
                r2 = qwen_chat(messages, stream=False)
                if not r2.ok:
                    head = ""
                    try: head = r2.text[:500]
                    except Exception: pass
                    log_event("upstream_error", {"when": "fallback", "status": r2.status_code, "head": head})
                    print(f"[Upstream FALLBACK ERROR] status={r2.status_code}\n{head}")
                    raise RuntimeError(f"fallback upstream {r2.status_code}")

                j = json.loads(r2.content.decode("utf-8", errors="replace"))
                ans = (j.get("choices", [{}])[0].get("message", {}) or {}).get("content") \
                      or j.get("output") \
                      or "(空响应)"
            except Exception as e2:
                ans = f"（上游错误：{str(e2)}）"

            for i in range(0, len(ans), 28):
                yield sse("delta", {"t": ans[i:i+28]}); time.sleep(0.02)
            yield sse("done", "")
            log_event("llm_stream_end", {"ok": False, "fallback": True})

    return Response(stream_with_context(gen()), content_type="text/event-stream; charset=utf-8")


# ========= 非流式（备份） =========
@app.route("/api/send", methods=["POST"])
def api_send_compat():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()

    displays = session.get("picks_display") or []
    actuals  = session.get("picks_actual") or []

    if not text and actuals:
        text = "请基于我刚刚附带的文件或图片，进行有用的解读、摘要与建议；如需明确目标，请先用一句话澄清后再回答。"

    strategy = get_conf()["strategy"]
    attach_msgs = build_attach_msgs(actuals)
    messages = build_messages(strategy, text, attach_msgs)
    
    # ★ 冲突检测：只在检测到用户明确澄清/纠正时才清空附件
    has_conflict = detect_conflict(text, attach_msgs)
    if has_conflict:
        session["picks_display"] = []
        session["picks_actual"]  = []
        log_event("conflict_detected_clear_attachments", {"text": text, "attach_count": len(actuals)})

    try:
        r = qwen_chat(messages, stream=False)
        if not r.ok:
            head = ""
            try: head = r.text[:500]
            except Exception: pass
            log_event("upstream_error", {"when": "send_non_stream", "status": r.status_code, "head": head})
            print(f"[Upstream NON-STREAM ERROR] status={r.status_code}\n{head}")
            raise RuntimeError(f"upstream {r.status_code}")

        j = json.loads(r.content.decode("utf-8", errors="replace"))
        ans = (j.get("choices", [{}])[0].get("message", {}) or {}).get("content") \
              or j.get("output") \
              or "(空响应)"
    except Exception as e:
        ans = f"（上游错误：{str(e)}）"

    # 非流式接口不在此处清空附件，由冲突检测统一处理
    return jsonify(ok=True, assistant_text=ans)


# ========= 前端行为日志 =========
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
