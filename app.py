# -*- coding: utf-8 -*-
"""
NoviceTrack 实验平台 - Flask 后端
- ModelScope（OpenAI 兼容 /chat/completions）Qwen3-VL-235B
- SSE 流式（逐字节→按行解码，修复中文乱码）
- 多附件：图片/文件 选择、任务 I 欺骗、任务 II 继承检测、任务 III 歧义检测
- 只发附件：自动生成可用上下文（文本抽样/图片内联或线索）
"""

import os
import re
import io
import time
import json
import uuid
import base64
import random
import mimetypes
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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
MODEL = os.getenv("MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")  # 视觉聊天模型

# 资源与日志
ASSETS_DIR = Path(os.getenv("ASSETS_DIR", "assets/candidates"))
LOG_DIR = Path(os.getenv("LOG_DIR", "data/logs"))
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

@app.before_request
def ensure_fresh_session_after_restart():
    """
    如果浏览器带来的 session 不是本次进程的 BOOT_ID，
    说明是上一次运行留下的，会在这里清空并初始化为新对话。
    """
    if session.get("_boot_id") != SERVER_BOOT_ID:
        # 清空所有旧状态（session_id、trial_id、附件选择、carry-over 等都会被重置）
        session.clear()
        # 标记为本次进程的 BOOT_ID，后续请求不会再触发清空
        session["_boot_id"] = SERVER_BOOT_ID
        # 初始化成默认配置，并开一个新的试次
        session["conf"] = DEFAULT_CONF.copy()
        session["trial_id"] = f"t_{int(time.time()*1000)}_{uuid.uuid4().hex[:4]}"
        # 其余键会在用到时按你原逻辑懒创建，比如 session_id()

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
DEFAULT_CONF = {"line": "松", "strategy": "B", "mode": "free"}  # 岚=A 自信纠正；松=B 谦逊澄清；雾=C 直接回答
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
    # 重置与附件/文本相关的 carry-over 状态
    session.pop("last_text_kw", None)
    session.pop("last_file_rel", None)
    session.pop("picks_display", None)
    session.pop("picks_actual", None)
    session.pop("unchanged_file_turns", None)
    return c


# ========= 日志 =========
def log_path() -> Path:
    return LOG_DIR / f"{session_id()}.json"

def read_events() -> List[Dict]:
    p = log_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception:
        return []

def append_events(evs: List[Dict]):
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


# ========= 资源/选择器 =========
def iter_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file()]
    files.sort(key=lambda x: x.as_posix().lower())
    return files

def human_size(n: int) -> str:
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024: return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.1f}TB"

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


# ========= 风格提示词 =========
SYSTEM_PROMPTS = {
    "A": "你是冷静专业的助手：表达简洁直接；当信息可能不一致时直说并给出合理结论与简要依据；信息充分则直接回答。",
    "B": "你是谦逊支持型助手：语气温和；当信息含糊或可能冲突时，先简短澄清或提供可选项，再继续回答；强调确认与透明。",
    "C": "你是中性直接型助手：围绕用户目标给出清晰步骤与答案；除非被要求，少做额外假设或纠偏提示。",
}


# ========= ModelScope 调用 =========
def qwen_chat(messages: List[Dict], stream: bool):
    """
    走 OpenAI 兼容 /chat/completions
    """
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
    return requests.post(url, headers=headers, json=data, stream=stream, timeout=(10, 300))


def build_messages(strategy: str, user_text: str, attach_msgs: List[Dict]) -> List[Dict]:
    """
    OpenAI 多模态格式：
    - 文本：{"type":"text","text":"..."}
    - 图片：{"type":"image_url","image_url":{"url":"data:...;base64,...."}}
    若模型非视觉或图片太大，则降级为文本线索。
    """
    sys = {"role": "system", "content": SYSTEM_PROMPTS.get(strategy, SYSTEM_PROMPTS["B"])}

    is_vision = any(k in MODEL.lower() for k in ["vision", "vl", "qwen3-vl"])

    # 没附件：普通文本
    if not attach_msgs:
        return [sys, {"role": "user", "content": user_text or "(未填写)"}]

    # 有附件：拼接为 content 数组
    parts = []
    if user_text:
        parts.append({"type": "text", "text": user_text})

    for m in attach_msgs:
        if m.get("type") == "image" and is_vision and m.get("b64"):
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{m.get('mime','image/jpeg')};base64,{m['b64']}"},
            })
        elif m.get("type") == "text":
            parts.append({
                "type": "text",
                "text": f"【附件文本摘录·{m.get('name','')}】\n{m.get('text','')}"
            })
        else:
            hint = f"【附件线索】名称：{m.get('name','')}；类型：{m.get('mime','')}；大小：{m.get('size','')}"
            parts.append({"type": "text", "text": hint})

    return [sys, {"role": "user", "content": parts}]


# ========= 文本/图片读取 =========
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


# ========= 关键词/检测 =========
_re_hanzi = re.compile(r"[\u4e00-\u9fff]+")
_re_en = re.compile(r"[A-Za-z]{3,}")
_re_num = re.compile(r"\d{2,}")
STOP_SET = set("的了呢啊吗吧是有在到和与及并并且或者而但如果以及关于针对一些一个一种如何怎么可以需要是否请请问麻烦谢谢".split())

def extract_keywords_from_text(text: str) -> List[str]:
    if not text: return []
    kws = []
    for seg in _re_hanzi.findall(text):
        if len(seg) >= 2 and seg not in STOP_SET:
            kws.append(seg)
    kws += [s.lower() for s in _re_en.findall(text)]
    kws += _re_num.findall(text)
    seen, out = set(), []
    for k in kws:
        if k not in seen:
            seen.add(k); out.append(k)
    return out

def extract_keywords_from_filename(name: str) -> List[str]:
    if not name: return []
    parts = _re_hanzi.findall(name)
    parts += [p.lower() for p in re.split(r"[^A-Za-z0-9]+", name) if len(p) >= 2]
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p); out.append(p)
    return out

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

AMBIG = ("这个","这款","那个","哪一个","哪个","这类","那类","这里的","它","其","该","上述","以下")

def has_ambiguous_phrase(text: str) -> bool:
    t = text or ""
    return any(p in t for p in AMBIG)

def detect_issue(mode: str, user_text: str,
                 displays: List[Dict], actuals: List[Dict],
                 last_text_kw: Optional[List[str]], last_file_rel: Optional[str]) -> Tuple[str, Dict]:
    if not (user_text or "").strip():
        return "none", {"text_kw": []}

    text_kw = extract_keywords_from_text(user_text)
    actual_kw = []
    for a in actuals:
        actual_kw += extract_keywords_from_filename(a["name"])
    display_kw = []
    for d in displays:
        display_kw += extract_keywords_from_filename(d["name"])

    sim_text_actual = jaccard(text_kw, actual_kw)
    sim_text_display = jaccard(text_kw, display_kw)

    if mode == "task_iii" and has_ambiguous_phrase(user_text):
        return "ambiguous", {"text_kw": text_kw}

    if mode == "task_ii":
        if last_file_rel and actuals and any(a.get("rel") == last_file_rel for a in actuals):
            if sim_text_actual < 0.15 and text_kw:
                last_kw = last_text_kw or []
                if jaccard(text_kw, last_kw) < 0.3:
                    return "carryover", {"text_kw": text_kw}

    if mode == "task_i":
        if displays and actuals:
            if {d.get("rel") for d in displays} != {a.get("rel") for a in actuals}:
                if sim_text_display > sim_text_actual + 0.15 and sim_text_actual < 0.2 and text_kw:
                    return "mismatch", {"text_kw": text_kw}

    if text_kw and actuals and sim_text_actual < 0.08:
        return "mismatch", {"text_kw": text_kw}

    return "none", {"text_kw": text_kw}


# ========= SSE 工具 =========
def sse(event: str, data) -> bytes:
    if not isinstance(data, str):
        payload = json.dumps(data, ensure_ascii=False)
    else:
        payload = json.dumps({"t": data}, ensure_ascii=False)
    return (f"event: {event}\n" f"data: {payload}\n\n").encode("utf-8")


# ========= 视图 =========
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


# ========= 选择器（多选） =========
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

    # 欺骗（任务 I）：把“实际上传”换成邻近
    actual = disp
    if get_conf()["mode"] == "task_i":
        neigh = []
        if by_index.get(index - 1): neigh.append(by_index[index - 1])
        if by_index.get(index + 1): neigh.append(by_index[index + 1])
        if neigh:
            actual = random.choice(neigh)

    displays = session.get("picks_display") or []
    actuals  = session.get("picks_actual") or []

    # 去重（避免连点叠入）
    if any(d.get("index") == disp["index"] for d in displays):
        log_event("pick_dup", {"display": disp})
        return jsonify(ok=True, display=disp, dup=True)

    displays.append(disp)
    actuals.append(actual)
    session["picks_display"] = displays
    session["picks_actual"]  = actuals

    log_event("pick", {"display": disp, "actual": actual, "deception": get_conf()["mode"] == "task_i"})
    return jsonify(ok=True, display=disp, dup=False)


@app.route("/api/remove_pick", methods=["POST"])
def api_remove_pick():
    data = request.get_json(force=True)
    idx = int(data["index"])
    displays = session.get("picks_display") or []
    actuals  = session.get("picks_actual") or []
    # 根据 display.index 删除匹配项
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
    """
    新对话：重置当前 trial，并清空与上下文相关的会话状态。
    不改变 line / mode（保持当前线路与模式不变）
    """
    # 开启一个全新的试次
    new_trial()

    # 清空与上下文/附件相关的服务端状态
    session["picks_display"] = []
    session["picks_actual"] = []
    session["last_text_kw"] = None
    session["last_file_rel"] = None
    session["unchanged_file_turns"] = None

    # 记一条日志，便于排查
    log_event("new_chat", {"reason": "user_click"})

    return jsonify(ok=True, trial_id=session["trial_id"])

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


# ========= 发送（SSE 流式）=========
@app.route("/api/send_stream", methods=["POST"])
def api_send_stream():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()

    displays = session.get("picks_display") or []
    actuals  = session.get("picks_actual") or []
    last_text_kw = session.get("last_text_kw")
    last_file_rel = session.get("last_file_rel")

    # 只发附件：给友好默认文案
    if not text and actuals:
        text = "请基于我刚刚附带的文件或图片，进行有用的解读、摘要与建议；如需明确目标，请先用一句话澄清后再回答。"

    issue_type, meta = detect_issue(get_conf()["mode"], text, displays, actuals, last_text_kw, last_file_rel)
    strategy = get_conf()["strategy"]

    attach_msgs = build_attach_msgs(actuals)
    messages = build_messages(strategy, text, attach_msgs)

    def gen():
        log_event("llm_stream_begin", {"text": text, "issue": issue_type, "meta": meta,
                                       "attach_count": len(actuals)})

        # B 组遇冲突先弹澄清
        if strategy == "B" and issue_type != "none":
            if issue_type in {"mismatch", "carryover"}:
                modal = {"title":"为确保准确，请选择你要咨询的对象：",
                         "options":["按文本继续","按文件继续","不确定，先帮我识别"]}
            else:
                modal = {"title":"你指的是哪一个？",
                         "options":["选项1","选项2","我不确定，先帮我标注/识别"]}
            yield sse("modal", modal)
            yield sse("done", "")
            log_event("llm_stream_end", {"reason":"modal"})
            return

        if strategy == "A" and issue_type in {"mismatch","carryover"}:
            yield sse("meta", {"toast":"检测到可能不一致，已按文件线索自动更正。"})

        try:
            r = qwen_chat(messages, stream=True)
            if not r.ok:
                raise RuntimeError(f"upstream {r.status_code}")

            # 逐字节缓冲→遇到换行再统一 UTF-8 解码，避免中文被切断
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
                        log_event("llm_stream_end", {"ok":True, "text_len":len(acc)})
                        session["last_text_kw"] = extract_keywords_from_text(text)
                        session["last_file_rel"] = (actuals[0].get("rel") if actuals else None)
                        return
                    try:
                        obj = json.loads(payload)
                        delta = obj["choices"][0]["delta"].get("content","")
                    except Exception:
                        delta = ""
                    if delta:
                        acc += delta
                        yield sse("delta", {"t": delta})

            yield sse("done", "")
            log_event("llm_stream_end", {"ok":True})

        except Exception as e:
            # 回退：一次性
            try:
                r2 = qwen_chat(messages, stream=False)
                j = json.loads(r2.content.decode("utf-8", errors="replace"))
                ans = j.get("choices",[{}])[0].get("message",{}).get("content","(空响应)")
            except Exception:
                ans = "(占位回答) 当前服务不可用，请稍后再试。"
            for ch in [ans[i:i+28] for i in range(0,len(ans),28)]:
                yield sse("delta", {"t": ch}); time.sleep(0.02)
            yield sse("done", "")
            log_event("llm_stream_end", {"ok":False,"fallback":True})

    return Response(stream_with_context(gen()),
                    content_type="text/event-stream; charset=utf-8")


# ========= 非流式（备份） =========
@app.route("/api/send", methods=["POST"])
def api_send_compat():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()

    displays = session.get("picks_display") or []
    actuals  = session.get("picks_actual") or []
    last_text_kw = session.get("last_text_kw")
    last_file_rel = session.get("last_file_rel")

    if not text and actuals:
        text = "请基于我刚刚附带的文件或图片，进行有用的解读、摘要与建议；如需明确目标，请先用一句话澄清后再回答。"

    issue_type, meta = detect_issue(get_conf()["mode"], text, displays, actuals, last_text_kw, last_file_rel)
    strategy = get_conf()["strategy"]

    attach_msgs = build_attach_msgs(actuals)
    messages = build_messages(strategy, text, attach_msgs)

    if strategy == "B" and issue_type != "none":
        if issue_type in {"mismatch","carryover"}:
            modal = {"title":"为确保准确，请选择你要咨询的对象：","options":["按文本继续","按文件继续","不确定，先帮我识别"]}
        else:
            modal = {"title":"你指的是哪一个？","options":["选项1","选项2","我不确定，先帮我标注/识别"]}
        return jsonify(ok=True, modal=modal)

    try:
        r = qwen_chat(messages, stream=False)
        j = json.loads(r.content.decode("utf-8", errors="replace"))
        ans = j.get("choices",[{}])[0].get("message",{}).get("content","(空响应)")
    except Exception:
        ans = "(占位回答) 当前服务不可用，请稍后再试。"

    toast = None
    if strategy == "A" and issue_type in {"mismatch","carryover"}:
        toast = "检测到可能不一致，已按文件线索自动更正。"

    return jsonify(ok=True, assistant_text=ans, toast=toast)


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
