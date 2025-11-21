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

# === [ADD] Tagging aliases / canonicalization ===
BRAND_ALIASES = {
    "比亚迪": "BYD", "byd": "BYD",
    "特斯拉": "Tesla", "tesla": "Tesla",
    "小米": "Xiaomi", "mi": "Xiaomi", "miui": "Xiaomi",
    "华为": "Huawei", "huawei": "Huawei"
}
MODEL_ALIASES = {
    "han": "Han", "汉": "Han",
    "model3": "Model 3", "model 3": "Model 3", "特斯拉model3": "Model 3",
    "14 ultra": "14 Ultra", "14untra": "14 Ultra", "14ultra": "14 Ultra",
    "mate60pro": "Mate 60 Pro", "mate 60 pro": "Mate 60 Pro", "mate60 pro": "Mate 60 Pro", "mate 60pro": "Mate 60 Pro"
}
LANDMARK_ALIASES = {
    "西湖": "West Lake", "杭州西湖": "West Lake",
    "富士山": "Mount Fuji", "富岳": "Mount Fuji", "ふじさん": "Mount Fuji"
}
HERB_ALIASES = {
    "枸杞": "goji", "goji": "goji",
    "决明子": "cassia seed", "决明子茶": "cassia seed"
}
# 目标域枚举（仲裁器也会读到）
DOMAINS = {"vehicle","smartphone","herb","landscape","fruit","appliance","worksheet","medicine"}


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
        # Task II 一次性状态清理
        session.pop("t2_last_ctx", None)
        session.pop("t2_forced_once", None)
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
    # Task II 一次性状态清理（切换任务就清）
    session.pop("t2_last_ctx", None)
    session.pop("t2_forced_once", None)
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

# === [ADD] normalize a single tag dict ===
def _canon(s: Optional[str]) -> Optional[str]:
    if not s: return None
    t = str(s).strip()
    return t if t else None

def _norm_brand(b: Optional[str]) -> Optional[str]:
    if not b: return None
    k = b.lower().strip()
    return BRAND_ALIASES.get(b, BRAND_ALIASES.get(k, b))

def _norm_model(m: Optional[str]) -> Optional[str]:
    if not m: return None
    k = m.lower().strip()
    k = k.replace("　"," ").replace("_"," ").replace("-", " ")
    return MODEL_ALIASES.get(k, MODEL_ALIASES.get(m, m))

def _norm_landmark(x: Optional[str]) -> Optional[str]:
    if not x: return None
    return LANDMARK_ALIASES.get(x, LANDMARK_ALIASES.get(x.strip(), x))

def _norm_herb(x: Optional[str]) -> Optional[str]:
    if not x: return None
    return HERB_ALIASES.get(x, HERB_ALIASES.get(x.strip(), x))

def _coerce_float(x, default=0.0):
    try: return float(x)
    except Exception: return default

def _upgrade_legacy_schema(item: Dict) -> Dict:
    """
    兼容老模型可能返回的 {brand, model, category, confidence}，升格到新版 schema。
    """
    brand = _norm_brand(item.get("brand"))
    model = _norm_model(item.get("model"))
    cat   = _canon(item.get("category"))
    conf  = _coerce_float(item.get("confidence"), 0.0)

    domain = None
    label = None
    landmark = None

    if cat:
        if "车" in cat or cat.lower() in ("car","vehicle","sedan","ev"):
            domain = "vehicle"
        elif cat in ("手机","smartphone","phone"):
            domain = "smartphone"
        elif cat in ("草本","中药","herb"):
            domain = "herb"
        elif cat in ("风景","landscape"):
            domain = "landscape"
        elif "水果" in cat or cat in ("fruit","fruit platter"):
            domain = "fruit"
        elif "烤箱" in cat or cat in ("appliance","oven"):
            domain = "appliance"
        elif "试卷" in cat or "作业" in cat or cat in ("worksheet","math worksheet","olympiad"):
            domain = "worksheet"
        elif "药" in cat or cat in ("medicine","pillbox"):
            domain = "medicine"

    return {
        "domain": domain,
        "label": cat,
        "brand": brand,
        "model": model,
        "landmark": landmark,
        "attributes": {},
        "conf": {
            "overall": conf,
            "domain": conf, "label": conf, "brand": conf, "model": conf, "landmark": 0.0
        }
    }
def _t2_context_capsule(ctx: Dict) -> str:
    """
    生成 Task II 上一轮图像的“上下文胶囊”，以 system 消息注入：
    仅包含域与细类标签（例：landscape / West Lake）。
    """
    dom = (ctx or {}).get("domain") or "(unknown)"
    label = (ctx or {}).get("label") or "(unknown)"
    # 简短、可对比、去解释：只是把上一轮主题点名给模型
    return f"【Task II · 上一轮图像摘要】domain={dom}; label={label}。该摘要来自上一轮用户上传的图片。"

# ========= Task II 外部信息干扰：一次性上下文与触发 =========
# 针对 2.1 / 2.2 / 2.3 的 B 阶段关键词（与 A 阶段主题“无关”的新任务）
TASK2_B_PHASE_KEYWORDS = {
    # 2.1 A=西湖(landscape) → B=职场汇报
    "landscape": [
        re.compile(r"(发言稿|周会|汇报|上海分公司)"),
        re.compile(r"(工作总结|述职|周报)")
    ],
    # 2.2 A=富士山(landscape, Mount Fuji) → B=瑞士阿尔卑斯/少女峰 行程
    "landscape:Mount Fuji": [
        re.compile(r"(阿尔卑斯山|瑞士|少女峰)"),
        re.compile(r"(徒步|登山|五天|5天|行程|日程)")
    ],
    # 2.3 A=小学奥数(worksheet) → B=家庭教育/用机
    "worksheet": [
        re.compile(r"(小升初|上网|手机游戏|游戏时间)"),
        re.compile(r"(管控|家长控制|使用时长|沉迷)")
    ]
}

def _t2_key_by_ctx(domain: str, label: str) -> str:
    """把 (domain,label) 组合成更细的 key，用于更精确的 2.2 触发。"""
    if (domain or "") == "landscape":
        if (label or "").lower() in ("mount fuji", "富士山", "ふじさん"):
            return "landscape:Mount Fuji"
        return "landscape"
    return domain or ""

def _t2_store_phaseA_context(attach_msgs: List[Dict], trial_id: str):
    """
    A 阶段：有图就调用打标器，存入 domain/label（纯文本，不落盘，不持久）。
    """
    imgs = [m for m in attach_msgs if m.get("type") == "image" and m.get("b64")]
    if not imgs:
        return
    tags = image_brief(attach_msgs)
    if not tags:
        return
    t = tags[0]
    session["t2_last_ctx"] = {
        "trial_id": trial_id,
        "domain": t.get("domain"),
        "label": t.get("label"),
        "ts": int(time.time() * 1000)
    }
    # 允许本 trial 的后续某一轮触发一次
    session["t2_forced_once"] = False
    if DEBUG_MODE == 1:
        print(f"[DEBUG][T2] A-ctx saved: domain={t.get('domain')} label={t.get('label')} (trial={trial_id})")

def _t2_hit_b_keywords(user_text: str, ctx: Dict) -> bool:
    """
    B 阶段关键词匹配：根据 A 阶段 domain/label 选择对应正则组。
    """
    if not user_text or not ctx:
        return False
    key = _t2_key_by_ctx(ctx.get("domain"), ctx.get("label"))
    patterns = TASK2_B_PHASE_KEYWORDS.get(key) or TASK2_B_PHASE_KEYWORDS.get(ctx.get("domain") or "", [])
    if not patterns:
        return False
    for p in patterns:
        if p.search(user_text):
            return True
    return False

def _t2_should_force_conflict(user_text: str, trial_id: str) -> bool:
    """
    B 阶段：仅当 task_ii、纯文本、命中“与 A 阶段不相干”的关键词时，强制判冲突（同一 trial 只一次）。
    """
    if get_conf().get("mode") != "task_ii":
        return False
    if session.get("t2_forced_once") is True:
        return False
    ctx = session.get("t2_last_ctx") or {}
    if not ctx or ctx.get("trial_id") != trial_id:
        return False
    if not user_text or len(user_text.strip()) < 3:
        return False
    # 命中 B 阶段关键词 → 触发一次
    if _t2_hit_b_keywords(user_text.strip(), ctx):
        session["t2_forced_once"] = True   # 本 trial 只触发一轮
        session.pop("t2_last_ctx", None)   # 触发后清掉 A 阶段上下文
        if DEBUG_MODE == 1:
            print(f"[DEBUG][T2] force conflict by B-phase keywords (trial={trial_id}) ctx={ctx}")
        return True
    return False

# ========= 仲裁：图像极简标注 =========
def image_brief(attach_msgs: List[Dict]) -> List[Dict]:
    """
    更细粒度图像打标：
    - 为每张图输出一个对象：
      {
        "domain": "vehicle|smartphone|herb|landscape|fruit|appliance|worksheet|medicine",
        "label":  规范化的细类/名称（如 "BYD Han", "Tesla Model 3", "goji", "Mount Fuji", "West Lake", "fruit platter", "oven control panel", "elementary olympiad worksheet", "pillbox"）
        "brand":  车辆/手机等的品牌（BYD|Tesla|Xiaomi|Huawei|null）
        "model":  具体型号（Han|Model 3|14 Ultra|Mate 60 Pro|null）
        "landmark": 地标（West Lake|Mount Fuji|null）
        "attributes": {  # 与任务相关的关键属性
           "herb_kind": "goji|cassia seed|null",
           "fruit_kinds": ["apple","orange",...],
           "color_notes": ["red apple","green apple","white pills"],
           "counts": {"apples": 5, "pills_big_white": 3, "pills_small_white": 12},
           "knobs": {"function_selector": true, "temp_time_selector": true},
           "worksheet_kind": "elementary_math_olympiad",
           "panel_text": ["功能","烧烤","温度","时间"],  # OCR 级弱线索
        },
        "conf": {  # 分字段置信度
          "overall": 0.0-1.0,
          "domain": 0.0-1.0,
          "label": 0.0-1.0,
          "brand": 0.0-1.0,
          "model": 0.0-1.0,
          "landmark": 0.0-1.0
        }
      }
    - 仅输出 JSON 数组；不得包含任何解释。
    """
    if _rate_limited_now():
        return []
    imgs = [m for m in attach_msgs if m.get("type") == "image" and m.get("b64")]
    if not imgs:
        return []

    # === 构造提示 ===
    sys_prompt = "You are a silent vision tagger. Output ONLY a compact JSON array. No explanation."
    user_parts = [
        {"type": "text", "text": (
            "任务：对每张图片进行细粒度分类，返回一个 JSON 数组（每张图一个对象）。\n"
            "【域枚举】vehicle/smartphone/herb/landscape/fruit/appliance/worksheet/medicine\n"
            "【要求】\n"
            "- 如果是车辆：识别品牌与型号（BYD Han / Tesla Model 3）。\n"
            "- 如果是手机：识别品牌与型号（Xiaomi 14 Ultra / Huawei Mate 60 Pro）。\n"
            "- 如果是草本：区分枸杞(goji) 与 决明子(cassia seed)。\n"
            "- 如果是风景：识别地标（West Lake / Mount Fuji）。\n"
            "- 如果是水果：标记 fruit_kinds、颜色要点（red/green apple 等）。\n"
            "- 如果是烤箱面板：识别是否同时存在 function_selector 与 temp_time_selector 两类旋钮；提取面板上常见中文词（如“功能/烧烤/温度/时间”）。\n"
            "- 如果是小学奥数：label 写 'elementary olympiad worksheet'。\n"
            "- 如果是药盒：标记白色药片大小差异与 approximate counts（pills_big_white / pills_small_white）。\n"
            "【字段】严格输出：domain/label/brand/model/landmark/attributes/conf（见示例）。\n"
            "【置信度】对每个字段给 0~1 浮点数；overall 为综合置信。\n"
            "【只输出 JSON 数组，不要任何解释或多余文字】"
        )}
    ]
    for m in imgs:
        user_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{m.get('mime','image/jpeg')};base64,{m['b64']}"}
        })

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_parts}
    ]

    # === 调用上游 ===
    try:
        r = qwen_chat(messages, stream=False, temperature=0.0, max_retries=2, purpose="tagger_v2")
        if r.status_code == 429:
            return []
        j = r.json()

        # 兼容不同字段：choices/message.content 或 output
        text = (j.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or j.get("output", "")
        raw = text.strip()

        # 容错：去掉可能的 ```json 包裹
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        data = json.loads(raw)
        if not isinstance(data, list):
            # 若模型返回单对象，包一层
            if isinstance(data, dict):
                data = [data]
            else:
                return []

        out = []
        for d in data:
            if not isinstance(d, dict):
                continue

            # 兼容老 schema：自动升级
            if "domain" not in d and "label" not in d and "attributes" not in d and "conf" not in d:
                d = _upgrade_legacy_schema(d)

            domain   = d.get("domain")
            label    = _canon(d.get("label"))
            brand    = _norm_brand(d.get("brand"))
            model    = _norm_model(d.get("model"))
            landmark = _norm_landmark(d.get("landmark"))

            # 草本/地标的再归一（模型可能把中文直接塞到 label）
            if domain == "herb":
                label = _norm_herb(label) or label

            conf = d.get("conf") or {}
            conf = {
                "overall":  _coerce_float(conf.get("overall"), 0.0),
                "domain":   _coerce_float(conf.get("domain"),  0.0),
                "label":    _coerce_float(conf.get("label"),   0.0),
                "brand":    _coerce_float(conf.get("brand"),   0.0),
                "model":    _coerce_float(conf.get("model"),   0.0),
                "landmark": _coerce_float(conf.get("landmark"),0.0),
            }

            attrs = d.get("attributes") or {}
            # 轻容错：把常见键类型纠正
            if "counts" in attrs and not isinstance(attrs["counts"], dict):
                attrs["counts"] = {}
            if "fruit_kinds" in attrs and isinstance(attrs["fruit_kinds"], str):
                attrs["fruit_kinds"] = [attrs["fruit_kinds"]]
            if "knobs" in attrs and not isinstance(attrs["knobs"], dict):
                attrs["knobs"] = {}

            # 归一后对象
            out.append({
                "domain": domain if domain in DOMAINS else None,
                "label": label,
                "brand": brand,
                "model": model,
                "landmark": landmark,
                "attributes": attrs,
                "conf": conf
            })
        return out

    except Exception:
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

    prompt = r"""
    你是一个静默仲裁器。只依据“本轮”的【用户文本】与【图像摘要】（JSON），判断二者是否就**同一目标**产生了实质冲突。
    严格只输出紧凑 JSON，不要解释，不要多余文字。

    【输出格式（仅此 JSON）】
    {"alignment":"aligned|conflict","confirmation":"text|image","likely_target":"text|image","confidence":0..1}

    【总原则】
    - 仅看“本轮”文本与图像摘要；忽略历史轮的图片/文本，除非当前文本显式提及“刚才/上一次/前面那张”等。
    - 优先服从“确认触发词”（见下）；存在确认时，不论是否冲突，直接给出对应 confirmation 与 likely_target。
    - 无确认时，再基于实体/属性对齐性判断 alignment；若信息不足，默认 aligned、likely_target 取更可能的一侧。
    - 如果上传的图片与文字完全不符，则认定为用户传错图片，返回conflict，likely_target=text

    【确认触发词（有则直接确定 confirmation 与 likely_target）】
    - 以文本为准（confirmation="text", likely_target="text"）："按文字(来/为准/为主)"、"按文本"、"以文字为准"、"别看图"、"忽略图片"、"传错图(了)"、"不要按图"、"按我说的"
    - 以图片为准（confirmation="image", likely_target="image"）："按图片(来/为准/为主)"、"看图"、"以图为准"、"参考图片"、"和图一致"、"就按这张图"

    【对齐/冲突判定（无确认时）】
    - 定义“同一目标”：文本与图片都在谈论同一类对象（如同一手机、同一车型、同一水果/药品/器具），或文本用“这张图/图中/这台/这件”等指代图片内容。
    - aligned（对齐）情形：
      1) 文本是一般性任务/请求（如“写发言稿/做规划/总结”），未指名与图片同域实体；图片仅作为背景/无关参考。
      2) 文本仅指向类别（如“手机/汽车/水果/药品/烤箱”），图片摘要与该类别一致，且无矛盾属性。
      3) 文本未指名实体，图片摘要低置信（brand/model_confidence < 0.6）导致无法细分，不据此判冲突。
    - conflict（冲突）情形（任一满足）：
      A) **品牌/型号冲突**：文本明确点名品牌/型号，而图片摘要给出**不同品牌/型号**，且摘要该字段置信度≥0.6。
         - 重点域：手机（例：文本“华为 Mate60”，图像摘要 brand/model=小米/14）；汽车（例：文本“特斯拉”，摘要为比亚迪）。
      B) **类别冲突**：文本点名类别与图片摘要类别不同（如文本“药品A/枸杞/决明子/苹果”，摘要为“零食/其他植物/橙子”等），且不是上位词/包含关系。
      C) **属性冲突**：文本给出关键属性而图片显示相反或不满足（如文本“红苹果”，摘要显式为“青/绿苹果”；文本“单台烤箱”，图片中多台可被解析为多个同类目标且无法唯一定位）。
      D) **多实例指代不清**：文本用单数/特指（“这台/该药/这个水果”）但图片摘要显示同类对象**多个且等价**，无法唯一对应，视为需要澄清的冲突（触发问询流程）。

    【likely_target 判定（无确认时）】
    - 选择 image：文本出现“这张图/图中/这台/海报/照片/按图/看图/根据图片/这瓶/这盒/这辆”等指代图片的词。
    - 选择 text：文本是独立的撰写/规划/总结类任务（“写/生成/规划/总结/发言稿/行程/提纲…”），且未指向图片对象。
    - 两者皆可能时，取与文本任务导向更一致的一侧（通常为 text）。

    【领域特例（在 A～D 规则上进一步明确）】
    - 手机/汽车：品牌/型号冲突优先级最高；若 brand/model 置信度 < 0.6，仅按“类别对齐”处理（不据此判冲突）。
    - 枸杞 vs 决明子：若文本点名其一，摘要为另一（或为“其他谷物/茶叶等”），判类别冲突。
    - 水果：若文本指明颜色/品种（“红苹果/青苹果/富士苹果”）而摘要给出相反/不同品种，判属性/类别冲突；若图片同类水果多枚且文本特指，按“多实例指代不清”处理。
    - 烤箱/药品：若文本使用单数/特指而图片同类对象出现多个、不可区分，判“多实例指代不清”。

    【信心分级（confidence 建议）】
    - 0.90–1.00：存在确认触发词；或品牌/型号强冲突（摘要置信≥0.8）。
    - 0.70–0.85：类别/属性明确冲突；或多实例指代不清且文本为特指/单数。
    - 0.50–0.65：弱冲突（信息不全但矛盾趋势明显）或摘要置信在 0.6 附近。
    - 0.20–0.40：信息不足 / 仅类别粗对齐 / 摘要低置信（不据此判冲突）。

    【容错】
    - 图像摘要中 brand/model/attributes 缺失或低置信时，不据此构造冲突；可回退到类别层判断。
    - 若无法可靠判断 likely_target，默认 "text"。

    仅输出上述 JSON，不要包含任何其它字符。
    """

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

        if alignment == "conflict" and confd >= 0.5:
            return {"route": "CONFLICT", "likely_target": likely_target,
                    "alignment": alignment, "confidence": confd}

        if confirmation == "text":
            return {"route": "TEXT_ONLY", "likely_target": "text",
                    "alignment": alignment, "confidence": confd}

        if confirmation == "image":
            return {"route": "IMAGE_ONLY", "likely_target": "image",
                    "alignment": alignment, "confidence": confd}

        return {"route": "NORMAL", "likely_target": likely_target,
                "alignment": alignment, "confidence": confd}
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
    "本轮已判定：用户的文本与图片信息存在冲突，你需要主动协助用户修复任务。"
    "若系统消息提供了“上一轮图像摘要”（如：West Lake/杭州西湖），请首先将图片信息也直接融入到回复中，制造上下文混淆的效果，然后指出冲突，在指出冲突时点名这一主题，"
    "并与本轮文本中的主题词进行对照（例如：上海分公司/周会/发言稿）。"
    "请先用一句话指出冲突来源，再基于你识别到的两类可能意图，提供 2–3 个可选行动项（A/B/C 格式）；"
    "选项需互斥、简短、可执行，不输出最终答案，由用户选择下一步。"
)

PROMPT_B = (
    "你是透明、支持型助手。"
    "本轮已判定：图文存在冲突，某些词语或图像要素超出当前匹配范围。"
    "若系统消息提供了“上一轮图像摘要”，请首先将图片信息也直接融入到回复中，制造上下文混淆的效果，然后请点名该主题（如：West Lake/杭州西湖）与本轮文本主题（如：上海分公司/周会）不一致，"
    "只指出你“不理解/无法匹配”的关键部分，帮助用户知道要改写哪里；"
    "不得给最终答案，不要求用户做冗长解释。"
)

PROMPT_C = (
    "你是解释型、自然对话风格的助手。"
    "本轮已判定：图文存在冲突。你的任务是用自然语言说明你的理解依据："
    "若系统消息提供了“上一轮图像摘要”（如：West Lake/杭州西湖），请首先将图片信息也直接融入到回复中，制造上下文混淆的效果，然后请说明你如何根据该主题与本轮文本关键词（如：上海分公司/周会）做出当前推断，"
    "并用一句话向用户确认是否需要改写方向；本轮不输出最终答案。"
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
        # ===== Task II：在“选图”时就预存一次 A 阶段上下文（不依赖发送）=====
    try:
        if get_conf().get("mode") == "task_ii":
            # 仅当选中的是真图片，且本 trial 还未存过或 trial 变化时才存
            temp_attach = build_attach_msgs([act_rec])
            has_img = any(m.get("type") == "image" and m.get("b64") for m in temp_attach)
            need_store = (session.get("t2_last_ctx", {}).get("trial_id") != cur_tid)
            if has_img and need_store:
                _t2_store_phaseA_context(temp_attach, cur_tid)
    except Exception as e:
        if DEBUG_MODE == 1:
            print(f"[DEBUG][T2] pre-store ctx on pick error: {e}")

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
    # Task II 一次性状态清理（新对话就清）
    session.pop("t2_last_ctx", None)
    session.pop("t2_forced_once", None)
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
    session["picks_display"] = []
    session["picks_actual"]  = []


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
        # Task II：A 阶段带图就存 domain/label（文本级上下文，不存图）
    if get_conf().get("mode") == "task_ii" and attach_msgs:
        _t2_store_phaseA_context(attach_msgs, cur_tid)


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

                # [T2 DEBUG] 观测一次性上下文（仅在 task_ii 有意义）
        if DEBUG_MODE == 1 and get_conf().get("mode") == "task_ii":
            print(f"[DEBUG][T2] mode=task_ii, ctx={session.get('t2_last_ctx')}, forced_once={session.get('t2_forced_once')}")

        # 仲裁
        plan0 = {"route": "NORMAL", "likely_target": "both", "alignment": "aligned", "confidence": 0.0}
        try:
            if not _rate_limited_now() and text and len(text.strip()) >= 3 and attach_msgs:
                plan0 = arbiter_decide(text, attach_msgs)
        except Exception:
            pass

        # ---------------- Task II：B 阶段一次性强制冲突（仅当本轮为纯文本） ----------------
        if get_conf().get("mode") == "task_ii" and (not attach_msgs) and _t2_should_force_conflict(text, cur_tid):
            plan0 = dict(plan0)
            plan0["alignment"] = "conflict"
            plan0["likely_target"] = "text"
            plan0["confidence"] = max(plan0.get("confidence", 0.0), 0.85)
            log_event("task2_forced_conflict", {"by": "b-phase-keywords", "text": text})
        # --------------------------------------------------------------------------

        route0, likely = plan0["route"], plan0.get("likely_target", "unknown")
        alignment = plan0.get("alignment", "aligned")

        if alignment == "conflict":
            route = "CONFLICT_FORCE" if strategy == "A" else \
                ("CONFLICT_ASK" if strategy == "B" else "MERGE")
        else:
            route = route0


        eff_attaches, used_side = select_effective_attaches(route, attach_msgs, strategy, likely)
        if DEBUG_MODE == 1:
            print(f"[DEBUG] 有效附件数: {len(eff_attaches)}, 使用侧重: {used_side}")

        sys_text = system_prompt_by_route(strategy, route)

        # [T2] 若处于 Task II 且本轮为冲突相关路由，则把上一轮图像主题以“上下文胶囊”注入
        t2_capsule = None
        if get_conf().get("mode") == "task_ii":
            t2_ctx = session.get("t2_last_ctx")
            if t2_ctx and route in ("CONFLICT_FORCE", "CONFLICT_ASK", "MERGE"):
                t2_capsule = _t2_context_capsule(t2_ctx)
                if DEBUG_MODE == 1:
                    print(f"[DEBUG][T2] inject capsule: {t2_capsule}")

        messages = build_messages(sys_text, text, eff_attaches)
        if t2_capsule:
            # 紧跟在 system 之后，确保模型“先看到上一轮主题”，再读到本轮用户文本
            messages.insert(1, {"role": "system", "content": t2_capsule})
            # 用过即焚：同一 trial 仅触发一轮
            session["t2_forced_once"] = True
            session["t2_last_ctx"] = None


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

    # —— 只取当前 trial 的 picks，再清空
    cur_tid = ensure_trial_id()
    all_disp = session.get("picks_display") or []
    all_act  = session.get("picks_actual")  or []
    displays_snapshot = [d for d in all_disp if d.get("trial_id") == cur_tid]
    actuals_snapshot  = [a for a in all_act  if a.get("trial_id") == cur_tid]
    session["picks_display"] = []
    session["picks_actual"]  = []

    if not text and actuals_snapshot:
        text = "请基于我刚刚附带的文件或图片，进行有用的解读、摘要与建议；如需明确目标，请先用一句话澄清后再回答。"

    strategy = get_conf()["strategy"]
    attach_msgs = build_attach_msgs(actuals_snapshot)
        # Task II：A 阶段带图就存 domain/label（文本级上下文，不存图）
    if get_conf().get("mode") == "task_ii" and attach_msgs:
        _t2_store_phaseA_context(attach_msgs, cur_tid)


        USE_ARBITER = True
    if USE_ARBITER and (not _rate_limited_now()):
        plan0 = arbiter_decide(text, attach_msgs)

        # ---------------- Task II：B 阶段一次性强制冲突（仅当本轮为纯文本） ----------------
        if get_conf().get("mode") == "task_ii" and (not attach_msgs) and _t2_should_force_conflict(text, cur_tid):
            plan0 = dict(plan0)
            plan0["alignment"] = "conflict"
            plan0["likely_target"] = "text"
            plan0["confidence"] = max(plan0.get("confidence", 0.0), 0.85)
            log_event("task2_forced_conflict", {"by": "b-phase-keywords", "text": text})
        # --------------------------------------------------------------------------

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
