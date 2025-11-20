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

from io import BytesIO
try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False


# ========= 基础配置 =========
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5010"))
DEBUG = os.getenv("DEBUG", "1") == "1"
DEBUG_MODE = 1  # 设置为 1 开启调试模式，设置为 0 关闭调试模式


# DashScope / OpenAI-兼容（阿里云）
# 说明：
# - 使用环境变量 DASHSCOPE_API_KEY 提供密钥（不要硬编码）
# - 基础地址固定为兼容模式，/chat/completions 会自动拼上
MS_API_KEY  = os.getenv("DASHSCOPE_API_KEY", "sk-d22d88924be5480294aa9091e8033437")  # ← 你的阿里云 key 放到这个 env
MS_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL       = os.getenv("MODEL", "qwen3-vl-plus")  # 例如：qwen3-vl-plus / qwen2.5-vl-72b-instruct 等

# （可选）启用“思考过程”，与 DashScope 兼容模式参数一致
DASHSCOPE_ENABLE_THINKING = os.getenv("DASHSCOPE_ENABLE_THINKING", "0") == "1"
try:
    DASHSCOPE_THINKING_BUDGET = int(os.getenv("DASHSCOPE_THINKING_BUDGET", "0") or "0")
except Exception:
    DASHSCOPE_THINKING_BUDGET = 0


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

THUMB_DIR = Path(os.getenv("THUMB_DIR", str(BASE_DIR / "assets" / "_thumbs")))
THUMB_DIR.mkdir(parents=True, exist_ok=True)


GRID_COLS_DEFAULT = 5
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
TEXT_EXTS = {".txt", ".md", ".json", ".csv", ".log"}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}
INLINE_IMAGE_LIMIT = 350 * 1024  # 仅小图内联 base64

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

def _wait_rate_cooldown(extra: float = 0.0, cap: float = 12.0):
    """
    等待当前限流窗口结束；可加额外等待(extra)，并用 cap 设一个最长等待上限，防止长时间阻塞。
    - 若当前未处于限流窗口，只睡 extra。
    - 若处于窗口内，睡 (窗口剩余 + extra)，但不超过 cap。
    """
    now = time.time()
    remain = 0.0
    if _rate_limited_now():
        remain = max(0.0, _RATE_LIMIT_UNTIL - now)
    wait = min(max(0.0, remain + max(0.0, extra)), max(0.0, cap))
    if wait > 0:
        if DEBUG_MODE == 1:
            print(f"[DEBUG] Cooldown sleep {wait:.1f}s (remain={remain:.1f}, extra={extra:.1f})")
        time.sleep(wait)

# ========= 会话/重启保护 =========
@app.before_request
def ensure_fresh_session_after_restart():
    if session.get("_boot_id") != SERVER_BOOT_ID:
        session.clear()
        session["_boot_id"] = SERVER_BOOT_ID
        session["conf"] = DEFAULT_CONF.copy()
        session["trial_id"] = f"t_{int(time.time()*1000)}_{uuid.uuid4().hex[:4]}"
        clear_history()  # 清空全局历史
        # 不再需要 session["chat_history"] 这个字段了



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


# ========= 会话历史（多轮记忆）=========
HISTORY_MAX_TURNS = 10  # 只保留最近 8 轮 user+assistant，对应 16 条 message
# 全局历史存储：key = session_id(), value = [ {"role": "...", "content": "..."} , ... ]
_GLOBAL_CHAT_HISTORY: Dict[str, List[Dict]] = {}
def get_history() -> List[Dict]:
    """
    从全局字典中取出当前 session 的历史。
    注意：不再依赖 session["chat_history"]。
    """
    sid = session_id()
    h = _GLOBAL_CHAT_HISTORY.get(sid)
    if not isinstance(h, list):
        h = []
        _GLOBAL_CHAT_HISTORY[sid] = h
    return h

def append_history_turn(user_text: str, assistant_text: str):
    """
    追加一轮 (user, assistant) 到历史，并进行截断。
    历史只存在于 _GLOBAL_CHAT_HISTORY 中。
    """
    h = get_history()
    h.append({"role": "user", "content": user_text or "(未填写)"})
    h.append({"role": "assistant", "content": assistant_text or ""})

    max_msgs = HISTORY_MAX_TURNS * 2  # 每轮两条
    if len(h) > max_msgs:
        # 就地截断，保留末尾 max_msgs 个
        del h[:-max_msgs]

    # 显式写回（虽然 h 是同一个 list，但语义更清晰）
    _GLOBAL_CHAT_HISTORY[session_id()] = h

def clear_history():
    """
    清空当前 session 的历史。
    """
    _GLOBAL_CHAT_HISTORY.pop(session_id(), None)

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
    # ★ 新增：切换任务即视为新对话，清空多轮历史
    clear_history()
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

def downscale_to_b64(path: Path, max_side: int = 512, max_bytes: int = 100*1024) -> Optional[str]:
    """
    将图片压缩成小尺寸 JPEG 并返回 base64（用于预览兜底）。
    若 Pillow 未安装或处理失败，返回 None。
    """
    try:
        from PIL import Image, ImageOps
    except Exception:
        return None
    try:
        import io, base64
        im = Image.open(path)
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side))
        q = 85
        while q >= 50:
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=q, optimize=True)
            b = buf.getvalue()
            if len(b) <= max_bytes or q <= 60:
                return base64.b64encode(b).decode("ascii")
            q -= 5
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
            "src": (f"/thumb/{rel.as_posix()}?w=360") if is_img else None,
        })
    return items


@app.route("/assets/<path:subpath>")
def serve_asset(subpath):
    target = (ASSETS_DIR / subpath).resolve()
    if not str(target).startswith(str(ASSETS_DIR.resolve())):
        abort(403)
    return send_from_directory(ASSETS_DIR, subpath)

@app.route("/thumb/<path:subpath>")
def serve_thumb(subpath):
    """
    生成并返回小尺寸 JPEG 缩略图；失败时回退到原文件。
    支持中文/空格文件名（浏览器会自动 URL 编码）。
    """
    target = (ASSETS_DIR / subpath).resolve()
    if not str(target).startswith(str(ASSETS_DIR.resolve())):
        abort(403)
    if (not target.exists()) or (not target.is_file()):
        abort(404)
    try:
        w = int(request.args.get("w", "360") or "360")
    except Exception:
        w = 360
    try:
        b64 = downscale_to_b64(target, max_side=max(64, min(2048, w)), max_bytes=120*1024)
        if not b64:
            # 无 Pillow 或压缩失败 → 直接回源
            return send_from_directory(ASSETS_DIR, subpath)
        data = base64.b64decode(b64.encode("ascii"))
        return Response(data, mimetype="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})
    except Exception:
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
    兼容 DashScope 的 OpenAI /chat/completions：
    - base: https://dashscope.aliyuncs.com/compatible-mode/v1
    - 路径: /chat/completions
    - Header: Authorization: Bearer <DASHSCOPE_API_KEY>
    - 支持 stream=True 走 SSE
    - 延续现有 429 退避与全局限流窗口
    """
    if not MS_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置，请在环境变量中提供。")

    url = f"{MS_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {MS_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }

    # DashScope 兼容模式的 payload 与 OpenAI 一致；可附加 enable_thinking / thinking_budget
    data = {
        "model": MODEL,
        "temperature": temperature,
        "messages": messages,
    }
    if stream:
        data["stream"] = True

    # （可选）启用“思考过程”（仅当设置了 env 时生效）
    if DASHSCOPE_ENABLE_THINKING:
        data["enable_thinking"] = True
        if DASHSCOPE_THINKING_BUDGET and DASHSCOPE_THINKING_BUDGET > 0:
            data["thinking_budget"] = DASHSCOPE_THINKING_BUDGET

    last_resp = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=data, stream=stream, timeout=(20, 300))
        except Exception as e:
            last_resp = None
            if attempt < max_retries - 1:
                wait = (0.8 * (2 ** attempt)) + random.uniform(0, 0.2)
                log_event("upstream_exception_retry", {
                    "purpose": purpose, "attempt": attempt+1, "wait": wait, "error": str(e)
                })
                time.sleep(wait)
                continue
            raise

        last_resp = resp

        # 非 429 直接返回；429 进入退避
        if resp.status_code != 429:
            return resp

        # 命中 429：指数退避 + 设置全局限流窗口
        wait_hdr = _retry_after_seconds(resp)
        wait = max(wait_hdr, 1.5 * (2 ** attempt)) + random.uniform(0, 0.5)
        _set_rate_limited(max(wait, 5.0))
        head = ""
        try:
            head = resp.text[:500]
        except Exception:
            pass
        log_event("rate_limited", {
            "purpose": purpose, "attempt": attempt+1, "status": resp.status_code,
            "wait": wait, "retry_after": wait_hdr, "head": head
        })
        if attempt < max_retries - 1:
            print(f"[RATE LIMIT] {purpose} attempt {attempt+1}, waiting {wait:.1f}s...")
            time.sleep(wait)
            continue

        # 最后一次也 429，设置 10s 冷却并返回
        _set_rate_limited(10.0)
        print(f"[RATE LIMIT] {purpose} max retries exhausted, setting 10s cooldown")
        return resp

    return last_resp



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

    # 限流窗口或无文本时跳过仲裁
    if _rate_limited_now():
        if DEBUG_MODE == 1:
            print("[DEBUG] Arbiter skipped - rate limited (pre-check)")
        return {"route": "NORMAL", "likely_target": "unknown", "alignment": "aligned", "confidence": 0.0}

    if not user_text or len(user_text.strip()) < 3:
        return {"route": "IMAGE_ONLY", "likely_target": "image", "alignment": "aligned", "confidence": 0.9}

    # —— 第一步：图片极简打标（可能命中 429 并设置限流窗口）
    img_tags = image_brief(attach_msgs)

    # tagger → arbiter 之间等待（先固定 1s，再根据限流窗口补等）
    time.sleep(1.0)
    if _rate_limited_now():
        # 若 tagger 设置了窗口，这里等待到窗口结束（再加 0.2s 抖动）
        _wait_rate_cooldown(extra=0.2, cap=12.0)

    # 仍在限流窗口：直接放弃仲裁，请求走 NORMAL，避免继续撞 429
    if _rate_limited_now():
        if DEBUG_MODE == 1:
            print("[DEBUG] Arbiter aborted - still rate limited after tagger")
        return {"route": "NORMAL", "likely_target": "text", "alignment": "aligned", "confidence": 0.0}

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
        r = qwen_chat(messages, stream=False, temperature=0.0, max_retries=1, purpose="arbiter")
        if r.status_code == 429:
            # 命中限流：不再继续，回退 NORMAL
            return {"route": "NORMAL", "likely_target": "text", "alignment": "aligned", "confidence": 0.0}
        j = r.json()
        raw = (j.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or j.get("output", "")
        info = json.loads(raw.strip())

        alignment = info.get("alignment", "aligned")
        confirmation = info.get("confirmation", "text")
        likely_target = info.get("likely_target", "text")
        confd = float(info.get("confidence") or 0.0)

        if DEBUG_MODE == 1:
            print(
                f"Debug Info - Alignment: {alignment}, Confirmation: {confirmation}, Likely Target: {likely_target}, Confidence: {confd}")

        if confirmation == "text":
            return {"route": "TEXT_ONLY", "likely_target": "text", "alignment": alignment, "confidence": confd}
        if confirmation == "image":
            return {"route": "IMAGE_ONLY", "likely_target": "image", "alignment": alignment, "confidence": confd}
        if alignment == "conflict" and confd >= 0.5:
            return {"route": "CONFLICT", "likely_target": likely_target, "alignment": alignment, "confidence": confd}
        return {"route": "NORMAL", "likely_target": likely_target, "alignment": alignment, "confidence": confd}

    except Exception:
        return {"route": "NORMAL", "likely_target": "text", "alignment": "aligned", "confidence": 0.0}


# ========= 提示词（仅在冲突时启用风格）=========
NEUTRAL_PROMPT = (
    "You are a helpful, concise assistant. "
    "You can use previous turns (both user and assistant messages) as context, "
    "but you should primarily answer based on the user's latest question. "
    "When the user later says that a previous image or statement was wrong "
    "(for example: '传错图了', '刚才那个不对', '其实是xxx'), "
    "you must treat the latest clarification as correct for future turns. "
    "Do not keep referring back to the mistaken image or interpretation, "
    "unless the user explicitly asks about that earlier wrong content "
    "(for example: '之前那张传错的图是什么')."
)


PROMPT_A = (
    "你是主动、清晰、结构化的助手。"
    "本轮已判定：用户的文本与图片信息存在冲突。你的目标是主动协助用户修复任务。"
    "请明确指出冲突，并基于你识别到的两类可能意图，提供2-3 个可选行动项（A/B/C 格式）；"
    "选项需互斥、简短、可执行，不输出最终答案，由用户选择下一步。"
)
PROMPT_B = (
    "你是透明、支持型助手。"
    "本轮已判定：图文存在冲突，其中某些词语或图像要素超出当前理解范围。"
    "请用**一句话（≤20字）**指出你无法识别或无法匹配的关键部分，帮助用户知道需要改写什么；"
    "不得给最终答案，不得要求用户详细解释，只点出你“不理解的词/元素”。"
)
PROMPT_C = (
    "你是解释型、自然对话风格的助手。"
    "本轮已判定：图文存在冲突。你的任务是用自然语言解释你当前的理解依据——"
    "即你是根据文本或图片中的哪些关键词/要素做出当前推断的；"
    "请用简短一句话描述你的理解逻辑，并向用户确认是否正确；本轮不输出最终答案。"
)

def system_prompt_by_route(strategy: str, route: str) -> str:
    if route == "NORMAL":
        return NEUTRAL_PROMPT
    elif route == "TEXT_ONLY":
        return (
            "You are a helpful, concise assistant. "
            "The user has indicated they want an answer based on their text description. "
            "Images are provided for reference, but prioritize the text description in your response. "
            "You may use previous turns as context, and if the user has corrected a previous mistake "
            "(e.g. wrong image or wrong model), always follow the latest clarification."
        )

    elif route == "IMAGE_ONLY":
        return NEUTRAL_PROMPT
    elif strategy == "A":
        return PROMPT_A
    elif strategy == "B":
        return PROMPT_B
    else:
        return PROMPT_C


# ========= 构造附件上下文 =========
def build_attach_msgs(actuals: List[Dict]) -> List[Dict]:
    out = []
    for a in actuals or []:
        rel = a.get("rel")
        p = (ASSETS_DIR / rel).resolve()
        if not p.exists():
            continue

        mime_guess = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        mime = a.get("mime") or mime_guess
        ext = p.suffix.lower()
        size = human_size(p.stat().st_size)

        if ext in IMAGE_EXTS:
            # 先尝试原图内联
            b64 = safe_base64_of_image(p)
            use_mime = mime
            # 超限则自动压缩到限制内
            if not b64:
                b64 = downscale_to_b64(p, max_bytes=INLINE_IMAGE_LIMIT)
                if b64:
                    use_mime = "image/jpeg"
            if b64:
                out.append({"type": "image", "name": p.name, "mime": use_mime, "size": size, "b64": b64})
            else:
                out.append({"type": "hint", "name": p.name, "mime": mime, "size": size})

        elif ext in TEXT_EXTS:
            txt = read_text_excerpt(p, 1200)
            if txt:
                out.append({"type": "text", "name": p.name, "mime": mime, "size": size, "text": txt})
            else:
                out.append({"type": "hint", "name": p.name, "mime": mime, "size": size})

        elif ext in PDF_EXTS or ext in DOCX_EXTS:
            out.append({"type": "hint", "name": p.name, "mime": mime, "size": size})

        else:
            out.append({"type": "hint", "name": p.name, "mime": mime, "size": size})

    return out



# ========= 根据路由裁剪附件 =========
def select_effective_attaches(route: str, attach_msgs: List[Dict], strategy: str, likely_target: str) -> Tuple[List[Dict], str]:
    """
    根据仲裁结果选择有效附件。
    注意：TEXT_ONLY 不再丢弃图片，而是保留图片但在 system prompt 中强调以文本为准。
    """
    if route == "TEXT_ONLY":
        # 修改：保留附件，但标记为"以文本为准"
        return attach_msgs, "text"
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
    """
    构造发往上游 Qwen 的 messages：
    messages = [system] + history + [current_user_with_optional_images]
    其中 history 只包含纯文本的 user/assistant 轮次，不再重复图片。
    """
    sys = {"role": "system", "content": system_text}
    is_vision = any(k in MODEL.lower() for k in ["vision", "vl", "qwen3-vl", "qwen3-vl-"])

    # 1. 取出历史（只存 text），来源于全局字典
    history = get_history()

    # 2. 构造“当前轮”的 user message（可能是纯文本，也可能包含 image_url 块）
    if not effective_attaches:
        cur_user = {"role": "user", "content": user_text or "(未填写)"}
    else:
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



        cur_user = {"role": "user", "content": parts}

    # 3. 拼接：system + history + 当前 user
    msgs = [sys]
    msgs.extend(history)
    msgs.append(cur_user)

    # ★★★ 就在这里加调试打印（只在 DEBUG_MODE == 1 时输出）★★★
    if DEBUG_MODE == 1:
        print("\n[DEBUG] ===== messages to Qwen =====")
        try:
            print(json.dumps(msgs, ensure_ascii=False, indent=2))
        except Exception:
            # 防止 json.dumps 出错时把程序搞挂，兜底直接 print 原对象
            print(msgs)

    return msgs



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

    # 当前 trial_id，用于隔离不同“新对话”的选择
    cur_tid = ensure_trial_id()

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

    # 去重逻辑保持不变（按 index 判重）
    if get_conf()["mode"] == "task_i":
        if any(a.get("index") == actual["index"] for a in actuals):
            log_event("pick_dup_actual", {"display": disp, "actual": actual})
            # 返回时也带上 trial_id，方便前端如需对比
            ret_disp = dict(disp);  ret_disp["trial_id"] = cur_tid
            ret_act  = dict(actual);ret_act["trial_id"] = cur_tid
            return jsonify(ok=True, display=ret_disp, actual=ret_act, dup=True)
    else:
        if any(d.get("index") == disp["index"] for d in displays):
            log_event("pick_dup_display", {"display": disp})
            ret_disp = dict(disp);  ret_disp["trial_id"] = cur_tid
            return jsonify(ok=True, display=ret_disp, dup=True)

    # —— 关键：写入时打上 trial_id
    disp_rec = dict(disp);    disp_rec["trial_id"] = cur_tid
    act_rec  = dict(actual);  act_rec["trial_id"]  = cur_tid

    displays.append(disp_rec)
    actuals.append(act_rec)
    session["picks_display"] = displays
    session["picks_actual"]  = actuals

    if DEBUG_MODE == 1:
        print(f"\n[DEBUG] ========== 图片选择 ==========")
        print(f"[DEBUG] 选中图片: {disp.get('name', 'unknown')}")
        print(f"[DEBUG] 当前附件总数: {len(actuals)} (trial={cur_tid})")

    log_event("pick", {"display": disp_rec, "actual": act_rec, "deception": get_conf()["mode"] == "task_i"})
    return jsonify(ok=True, display=disp_rec, actual=act_rec, dup=False)



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
    clear_history()  # ★ 新增
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

    # —— 只取“当前 trial”的 picks，避免上轮残留
    cur_tid = ensure_trial_id()
    all_disp = session.get("picks_display") or []
    all_act  = session.get("picks_actual")  or []

    displays_snapshot = [d for d in all_disp if d.get("trial_id") == cur_tid]
    actuals_snapshot  = [a for a in all_act  if a.get("trial_id") == cur_tid]

    # ★ 新增：记录一次“语义上的发送事件”
    #    用于后续计算：
    #    - 快速重发模式（和上一轮 llm_stream_end.ts 做差）
    #    - 否定/澄清关键词频率（按轮统计 text 中关键词）
    log_event("user_send", {
        "text": text,
        "has_attach": bool(actuals_snapshot),
        "attach_count": len(actuals_snapshot),
    })

    # 立刻清空（防止后续再叠加）
    # 发送后清理：
    # - 非任务II：全部清空（维持原行为）
    # - 任务II：只保留“当前试次”的图片，其它类型清空（实现“图片跨轮有效”）
    if get_conf()["mode"] == "task_ii":
        keep_d, keep_a = [], []
        for d, a in zip(all_disp, all_act):
            if d.get("trial_id") == cur_tid and d.get("is_image"):
                keep_d.append(d);
                keep_a.append(a)
        session["picks_display"] = keep_d
        session["picks_actual"] = keep_a
    else:
        session["picks_display"] = []
        session["picks_actual"] = []

    if DEBUG_MODE == 1:
        print(f"\n[DEBUG] ========== 新消息 ==========")
        print(f"[DEBUG] 用户文本: {text[:100]}...")
        print(f"[DEBUG] 附件数量(快照): {len(actuals_snapshot)} (trial={cur_tid})")
        if actuals_snapshot:
            for i, a in enumerate(actuals_snapshot):
                print(f"[DEBUG] 附件 {i + 1}: {a.get('name', 'unknown')}")

    # 仅附件默认文案
    if not text and actuals_snapshot:
        text = "请基于我刚刚附带的文件或图片，进行有用的解读、摘要与建议；如需明确目标，请先用一句话澄清后再回答。"

    # ★ 新增：本轮要写入历史的 user 文本（包含上面的默认文案）
    user_text_for_history = text or "(未填写)"

    strategy = get_conf()["strategy"]
    attach_msgs = build_attach_msgs(actuals_snapshot)

    if DEBUG_MODE == 1:
        print(f"[DEBUG] 构建的附件消息数: {len(attach_msgs)}")
        for i, am in enumerate(attach_msgs):
            print(f"[DEBUG] 附件消息 {i + 1} 类型: {am.get('type', 'unknown')}")

    def gen():
        # —— 首包：把预览发给前端
        previews = []
        for it in actuals_snapshot:
            rel = it.get("rel")
            name = it.get("name")
            is_image = bool(it.get("is_image"))
            pv = {"name": name, "is_image": is_image, "src": None, "b64": None}
            if rel and is_image:
                pv["src"] = f"/thumb/{rel}?w=360"
                try:
                    p = (ASSETS_DIR / rel).resolve()
                    b64 = downscale_to_b64(p, max_bytes=90 * 1024, max_side=512)
                    if b64:
                        pv["b64"] = f"data:image/jpeg;base64,{b64}"
                except Exception:
                    pass
            previews.append(pv)
        yield sse("meta", {"toast": "已接收，开始处理...", "previews": previews})

        # 仲裁
        plan0 = {"route": "NORMAL", "likely_target": "both", "alignment": "aligned", "confidence": 0.0}
        try:
            if not _rate_limited_now() and text and len(text.strip()) >= 3 and attach_msgs:
                plan0 = arbiter_decide(text, attach_msgs)
        except Exception:
            pass

        route0, likely = plan0["route"], plan0.get("likely_target", "unknown")
        if DEBUG_MODE == 1:
            print(f"[DEBUG] Arbiter result - route: {route0}, likely_target: {likely}, confidence: {plan0.get('confidence', 0)}")

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
        if DEBUG_MODE == 1:
            print(f"[DEBUG] 有效附件数: {len(eff_attaches)}, 使用侧重: {used_side}")

        sys_text = system_prompt_by_route(strategy, route)
        messages = build_messages(sys_text, text, eff_attaches)

        log_event("llm_stream_begin", {"text": text, "attach_count": len(attach_msgs),
                                       "plan": {"route": route, "likely": likely}})

        # （后续逻辑与原函数相同）
        try:
            r = qwen_chat(messages, stream=True,
                          temperature=0.3 if route in ("NORMAL", "TEXT_ONLY", "IMAGE_ONLY") else 0.2,
                          max_retries=1, purpose="main_stream")
            if (r is None) or (not r.ok):
                if r and r.status_code == 429:
                    msg = "系统当前请求过于频繁，请稍后再试（约 10 秒后恢复）"
                    for i in range(0, len(msg), 10):
                        yield sse("delta", {"t": msg[i:i + 10]}); time.sleep(0.02)
                    yield sse("done", "")
                    log_event("llm_stream_end", {"ok": False, "reason": "rate_limit_429"})
                    return

                # 兜底
                if _rate_limited_now():
                    msg = "系统当前请求过于频繁，请稍后再试（约 10 秒后恢复）"
                    for i in range(0, len(msg), 10):
                        yield sse("delta", {"t": msg[i:i + 10]}); time.sleep(0.02)
                    yield sse("done", "")
                    log_event("llm_stream_end", {"ok": False, "reason": "pre_fallback_rate_limited"})
                    return

                r2 = qwen_chat(messages, stream=False,
                               temperature=0.3 if route in ("NORMAL", "TEXT_ONLY", "IMAGE_ONLY") else 0.2,
                               max_retries=1, purpose="main_fallback")
                if (r2 is None) or (not r2.ok):
                    if r2 and r2.status_code == 429:
                        msg = "系统当前请求过于频繁，请稍后再试（约 10 秒后恢复）"
                        for i in range(0, len(msg), 10):
                            yield sse("delta", {"t": msg[i:i + 10]}); time.sleep(0.02)
                        yield sse("done", "")
                        log_event("llm_stream_end", {"ok": False, "reason": "fallback_429"})
                        return
                    head = ""
                    try: head = r2.text[:500]
                    except Exception: pass
                    log_event("upstream_error", {"when": "fallback", "status": getattr(r2, "status_code", -1), "head": head})
                    msg = "服务暂时不可用，请稍后重试"
                    for i in range(0, len(msg), 10):
                        yield sse("delta", {"t": msg[i:i + 10]}); time.sleep(0.02)
                    yield sse("done", "")
                    return

                j = r2.json()
                ans = (j.get("choices", [{}])[0].get("message", {}) or {}).get("content") \
                      or j.get("output") or "(空响应)"
                # ★ 写入历史
                append_history_turn(user_text_for_history, ans)


                for i in range(0, len(ans), 28):
                    yield sse("delta", {"t": ans[i:i + 28]});
                    time.sleep(0.02)
                yield sse("done", "")
                log_event("llm_stream_end", {"ok": True, "text_len": len(ans), "mode": "non_stream_fallback"})
                return

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
                    bbuf = bbuf[pos + 1:]
                    if not line_bytes:
                        continue
                    line = line_bytes.decode("utf-8", errors="replace")
                    if line.startswith("event:"):
                        continue
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        # ★ 把本轮 (user, assistant) 写入历史
                        append_history_turn(user_text_for_history, acc)


                        yield sse("done", "")
                        log_event("llm_stream_end", {"ok": True, "text_len": len(acc), "mode": "stream"})
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

            # ★ 也写入历史
            append_history_turn(user_text_for_history, acc)

            yield sse("done", "")
            log_event("llm_stream_end", {"ok": True, "text_len": len(acc), "mode": "stream_no_done"})

        except Exception as e:
            log_event("stream_exception", {"error": str(e)})
            msg = f"（上游错误：{str(e)}）"
            for i in range(0, len(msg), 28):
                yield sse("delta", {"t": msg[i:i + 28]}); time.sleep(0.02)
            yield sse("done", "")

    return Response(stream_with_context(gen()), content_type="text/event-stream; charset=utf-8")






# ========= 非流式（备份） =========
@app.route("/api/send", methods=["POST"])
def api_send_compat():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()

    # —— 只取当前 trial 的 picks
    cur_tid = ensure_trial_id()
    all_disp = session.get("picks_display") or []
    all_act = session.get("picks_actual") or []
    displays_snapshot = [d for d in all_disp if d.get("trial_id") == cur_tid]
    actuals_snapshot = [a for a in all_act if a.get("trial_id") == cur_tid]

    # 发送后清理（逻辑同流式端点）
    if get_conf()["mode"] == "task_ii":
        keep_d, keep_a = [], []
        for d, a in zip(all_disp, all_act):
            if d.get("trial_id") == cur_tid and d.get("is_image"):
                keep_d.append(d);
                keep_a.append(a)
        session["picks_display"] = keep_d
        session["picks_actual"] = keep_a
    else:
        session["picks_display"] = []
        session["picks_actual"] = []

    if not text and actuals_snapshot:
        text = "请基于我刚刚附带的文件或图片，进行有用的解读、摘要与建议；如需明确目标，请先用一句话澄清后再回答。"

    strategy = get_conf()["strategy"]
    attach_msgs = build_attach_msgs(actuals_snapshot)

    USE_ARBITER = True
    if USE_ARBITER and (not _rate_limited_now()):
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
    else:
        route = "NORMAL"
        likely = "both"

    eff_attaches, used_side = select_effective_attaches(route, attach_msgs, strategy, likely)
    sys_text = system_prompt_by_route(strategy, route)
    messages = build_messages(sys_text, text, eff_attaches)

    try:
        r = qwen_chat(messages, stream=False,
                      temperature=0.3 if route in ("NORMAL","TEXT_ONLY","IMAGE_ONLY") else 0.2,
                      max_retries=1, purpose="main_non_stream")
        if (r is None) or (not r.ok):
            if r and r.status_code == 429:
                ans = "系统当前请求过于频繁，请稍后再试（约 10 秒后恢复）"
                return jsonify(ok=True, assistant_text=ans)
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

    # ★ 把这轮对话写入历史（非流式兜底路径）
    append_history_turn(text or "(未填写)", ans)

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
