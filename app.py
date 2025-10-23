# -*- coding: utf-8 -*-
"""
NoviceTrack 实验平台 - Flask 后端（SSE-JSON 修复版）
- 单页应用 API（含 SSE 流式）
- 任务 I/II/III 的通用检测（无领域词）
- 线路切换（岚/松/雾 -> A/B/C）
- 候选选择器 + 任务 I 邻居欺骗
- 全量行为日志
- 兼容 /api/send（非流式回退）
"""

import os
import re
import time
import json
import uuid
import random
import mimetypes
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import requests
from flask import (
    Flask, render_template, request, jsonify, session,
    send_from_directory, abort, Response, stream_with_context
)

# ========= 配置 =========
try:
    from config import (
        HOST, PORT, DEBUG,
        AIECNU_API_KEY, AIECNU_BASE_URL, MODEL,
        ASSETS_DIR, LOG_DIR, GRID_COLS, SHOW_HIDDEN, IMAGE_EXTS
    )
except Exception:
    HOST = "0.0.0.0"
    PORT = 5010
    DEBUG = True
    AIECNU_API_KEY = os.getenv("AIECNU_API_KEY", "YOUR_KEY")
    AIECNU_BASE_URL = os.getenv("AIECNU_BASE_URL", "http://127.0.0.1:3000/v1")
    MODEL = os.getenv("MODEL", "gpt-4.1")
    ASSETS_DIR = Path("assets/candidates")
    LOG_DIR = Path("data/logs")
    GRID_COLS = 3
    SHOW_HIDDEN = False
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

ASSETS_DIR = Path(ASSETS_DIR)
LOG_DIR = Path(LOG_DIR)

app = Flask(__name__, static_url_path="/static", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "novicetrack-secret")


# ========= 通用响应头（避免旧 JS 缓存） =========
@app.after_request
def add_no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    # 关键：让 Nginx 等反代不要缓冲 SSE
    resp.headers["X-Accel-Buffering"] = "no"
    # 建议：保持长连接
    resp.headers["Connection"] = "keep-alive"
    return resp


@app.route("/healthz")
def healthz():
    return "ok", 200


# ========= 会话/试次 =========
DEFAULT_CONF = {"line": "松", "strategy": "B", "mode": "free"}  # 岚/松/雾 -> A/B/C
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
    session.pop("last_text_kw", None)
    session.pop("last_file_rel", None)
    session.pop("last_pick_display", None)
    session.pop("last_pick_actual", None)
    session.pop("unchanged_file_turns", None)
    return c


# ========= 日志 =========
def log_path() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{session_id()}.json"

def read_events() -> List[Dict]:
    p = log_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception:
        return []

def append_events(events: List[Dict]):
    arr = read_events()
    arr.extend(events)
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


# ========= 候选素材 =========
def iter_files(root: Path) -> List[Path]:
    files = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if p.is_file():
            if not SHOW_HIDDEN and p.name.startswith("."):
                continue
            files.append(p)
    files.sort(key=lambda x: x.as_posix().lower())
    return files

def human_size(n: int) -> str:
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f}{u}"
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
            "row": idx // GRID_COLS,
            "col": idx % GRID_COLS,
        })
    return items

@app.route("/assets/<path:subpath>")
def serve_asset(subpath):
    target = (ASSETS_DIR / subpath).resolve()
    if not str(target).startswith(str(ASSETS_DIR.resolve())):
        abort(403)
    return send_from_directory(ASSETS_DIR, subpath)


# ========= 风格提示词（仅风格，不绑具体领域） =========
SYSTEM_PROMPTS = {
    "A": "你是冷静专业的助手：表达简洁直接；当信息可能不一致时直说并给出合理结论与简要依据；信息充分则直接回答。",
    "B": "你是谦逊支持型助手：语气温和；当信息含糊或可能冲突时，先简短澄清或提供可选项，再继续回答；强调确认与透明。",
    "C": "你是中性直接型助手：围绕用户目标给出清晰步骤与答案；除非被要求，少做额外假设或纠偏提示。",
}

def call_llm(strategy: str, user_text: str, file_hint: str, stream: bool = False):
    url = f"{AIECNU_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {AIECNU_API_KEY}",
        # ↓↓↓ 这两行是关键（仅对 stream=True 真正起作用）
        "Accept": "text/event-stream" if stream else "application/json",
        "Accept-Encoding": "identity",  # 关闭 gzip/deflate，避免缓冲
    }
    data = {
        "model": MODEL,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS.get(strategy, SYSTEM_PROMPTS["B"])},
            {"role": "user",
             "content": f"问题：{user_text or '(未填写)'}。\n可用线索（文件/图片等）：{file_hint or '无'}。\n保持风格，不要限定具体领域。"}
        ],
    }
    if stream:
        data["stream"] = True
    # 建议：分别设置连接/读取超时；读取长一点，避免断流
    r = requests.post(url, headers=headers, json=data, stream=stream, timeout=(10, 300))
    return r



# ========= 关键词抽取（通用） =========
_re_hanzi = re.compile(r"[\u4e00-\u9fff]+")
_re_en = re.compile(r"[A-Za-z]{3,}")
_re_num = re.compile(r"\d{2,}")
STOP_SET = set("的了呢啊吗吧是有在到和与及并并且或者而但如果以及关于针对一些一个一种如何怎么可以需要是否请请问麻烦谢谢".split())

def extract_keywords_from_text(text: str) -> List[str]:
    if not text:
        return []
    kws = []
    for seg in _re_hanzi.findall(text):
        if len(seg) >= 2 and seg not in STOP_SET:
            kws.append(seg)
    for seg in _re_en.findall(text):
        kws.append(seg.lower())
    for seg in _re_num.findall(text):
        kws.append(seg)
    seen = set(); out = []
    for k in kws:
        if k not in seen:
            seen.add(k); out.append(k)
    return out

def extract_keywords_from_filename(name: str) -> List[str]:
    if not name:
        return []
    parts = _re_hanzi.findall(name)
    parts += [p.lower() for p in re.split(r"[^A-Za-z0-9]+", name) if len(p) >= 2]
    seen = set(); out = []
    for p in parts:
        if p and p not in seen:
            seen.add(p); out.append(p)
    return out

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union


# ========= 任务通用检测 =========
AMBIG = ("这个", "这款", "那个", "哪一个", "哪个", "这类", "那类", "这里的", "它", "其", "该", "上述", "以下")

def has_ambiguous_phrase(text: str) -> bool:
    t = text or ""
    return any(p in t for p in AMBIG)

def detect_issue(mode: str, user_text: str,
                 display_item: Optional[Dict], actual_item: Optional[Dict],
                 last_text_kw: Optional[List[str]], last_file_rel: Optional[str]) -> Tuple[str, Dict]:
    text_kw = extract_keywords_from_text(user_text)
    actual_kw = extract_keywords_from_filename(actual_item["name"]) if actual_item else []
    display_kw = extract_keywords_from_filename(display_item["name"]) if display_item else []

    sim_text_actual = jaccard(text_kw, actual_kw)
    sim_text_display = jaccard(text_kw, display_kw)

    # 任务 III：语言歧义
    if mode == "task_iii" and has_ambiguous_phrase(user_text):
        return "ambiguous", {"text_kw": text_kw}

    # 任务 II：上下文遗留（文件未变 + 文本转向）
    if mode == "task_ii":
        if last_file_rel and actual_item and last_file_rel == actual_item.get("rel"):
            if sim_text_actual < 0.15 and text_kw:
                last_kw = last_text_kw or []
                if jaccard(text_kw, last_kw) < 0.3:
                    return "carryover", {"text_kw": text_kw}

    # 任务 I：显示与实际不一致
    if mode == "task_i":
        if display_item and actual_item and display_item.get("rel") != actual_item.get("rel"):
            if sim_text_display > sim_text_actual + 0.15 and sim_text_actual < 0.2 and text_kw:
                return "mismatch", {"text_kw": text_kw}

    # 兜底：文本与实际几乎不相关
    if text_kw and actual_item and sim_text_actual < 0.08:
        return "mismatch", {"text_kw": text_kw}

    return "none", {"text_kw": text_kw}


# ========= SSE 工具（JSON 单行 + UTF-8） =========
def sse(event: str, data) -> bytes:
    """
    统一把 payload 写成一行 JSON，避免多行 data 帧导致前端逐行解析出错；
    同时返回 UTF-8 bytes，避免中文乱码。
    """
    if not isinstance(data, str):
        payload = json.dumps(data, ensure_ascii=False)
    else:
        payload = json.dumps({"t": data}, ensure_ascii=False)
    return (f"event: {event}\n" f"data: {payload}\n\n").encode("utf-8")


def chunk_text(s: str, n: int = 28):
    for i in range(0, len(s), n):
        yield s[i:i+n]


# ========= 视图 =========
@app.route("/")
def index():
    ensure_trial_id()
    # 你的 index.html 已经准备了 __BOOT__
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


# ========= 选择器 =========
@app.route("/api/picker_list")
def api_picker_list():
    items = build_picker_items()
    log_event("picker_list", {"count": len(items)})
    return jsonify(ok=True, items=items, cols=GRID_COLS)

@app.route("/api/pick", methods=["POST"])
def api_pick():
    data = request.get_json(force=True)
    display_index = int(data["index"])
    items = build_picker_items()
    display_item = next((x for x in items if x["index"] == display_index), None)
    if not display_item:
        return jsonify(ok=False, error="not found"), 404

    actual_item = display_item
    if get_conf()["mode"] == "task_i":
        r, c = display_item["row"], display_item["col"]
        cand = []
        if r > 0:  # 上
            cand.append(next((x for x in items if x["row"] == r-1 and x["col"] == c), None))
        cand.append(next((x for x in items if x["row"] == r+1 and x["col"] == c), None))  # 下
        if c > 0:  # 左
            cand.append(next((x for x in items if x["row"] == r and x["col"] == c-1), None))
        cand.append(next((x for x in items if x["row"] == r and x["col"] == c+1), None))  # 右
        cand = [x for x in cand if x]
        if cand:
            actual_item = random.choice(cand)

    session["last_pick_display"] = display_item
    session["last_pick_actual"] = actual_item

    if session.get("last_file_rel") == (actual_item.get("rel") if actual_item else None):
        session["unchanged_file_turns"] = (session.get("unchanged_file_turns") or 0) + 1
    else:
        session["unchanged_file_turns"] = 1

    log_event("pick", {
        "display": {k: display_item[k] for k in ("index","name","rel","row","col")},
        "actual":  {k: actual_item[k]  for k in ("index","name","rel","row","col")},
        "deception": get_conf()["mode"] == "task_i"
    })
    return jsonify(ok=True, display=display_item)


# ========= 发送（SSE 流式） =========
@app.route("/api/send_stream", methods=["POST"])
def api_send_stream():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()

    display = session.get("last_pick_display")
    actual = session.get("last_pick_actual")
    last_text_kw = session.get("last_text_kw")
    last_file_rel = session.get("last_file_rel")

    issue_type, meta = detect_issue(get_conf()["mode"], text, display, actual, last_text_kw, last_file_rel)
    strategy = get_conf()["strategy"]
    hint = actual["rel"] if actual else "无"

    def gen():
        log_event("llm_stream_begin", {"text": text, "issue": issue_type, "meta": meta})

        # B：先澄清
        if strategy == "B" and issue_type != "none":
            if issue_type in {"mismatch", "carryover"}:
                modal = {"title": "为确保准确，请选择你要咨询的对象：",
                         "options": ["按文本继续", "按文件继续", "不确定，先帮我识别"]}
            else:
                modal = {"title": "你指的是哪一个？",
                         "options": ["选项1", "选项2", "我不确定，先帮我标注/识别"]}
            yield sse("modal", modal)
            yield sse("done", "")
            log_event("llm_stream_end", {"reason": "modal"})
            return

        # A：先提示
        if strategy == "A" and issue_type in {"mismatch", "carryover"}:
            yield sse("meta", {"toast": "检测到可能不一致，已按文件线索自动更正。"})

        # 上游流式
        try:
            r = call_llm(strategy, text, hint, stream=True)
            acc = ""
            if not r.ok:
                raise RuntimeError(f"upstream {r.status_code}")
            for raw in r.iter_lines(decode_unicode=False):
                if not raw:
                    continue
                try:
                    line = raw.decode("utf-8", errors="replace")
                except Exception:
                    # 理论到不了这里，但留兜底
                    line = str(raw, "utf-8", errors="replace")

                if not line.startswith("data: "):
                    continue

                msg = line[6:].strip()
                if msg == "[DONE]":
                    break

                try:
                    obj = json.loads(msg)
                    delta = obj["choices"][0]["delta"].get("content", "")
                except Exception:
                    delta = ""

                if delta:
                    acc += delta
                    yield sse("delta", {"t": delta})

            yield sse("done", "")
            log_event("llm_stream_end", {"ok": True, "text_len": len(acc)})
        except Exception:
            # 回退：非流式一次性 + 本地分片
            try:
                r2 = call_llm(strategy, text, hint, stream=False)
                try:
                    j = json.loads(r2.content.decode("utf-8", errors="replace"))
                except Exception:
                    j = {}

                ans_text = j.get("choices", [{}])[0].get("message", {}).get("content", "(空响应)")
            except Exception:
                ans_text = f"(本地占位响应) 策略{strategy}：问题={text!r}；线索={hint!r}"
            for ch in chunk_text(ans_text, 28):
                yield sse("delta", {"t": ch}); time.sleep(0.02)
            yield sse("done", "")
            log_event("llm_stream_end", {"ok": False, "fallback": True, "text_len": len(ans_text)})

        # 更新记忆
        session["last_text_kw"] = extract_keywords_from_text(text)
        session["last_file_rel"] = actual.get("rel") if actual else None

    return Response(stream_with_context(gen()),
                    content_type="text/event-stream; charset=utf-8")


# ========= 兼容非流式 =========
@app.route("/api/send", methods=["POST"])
def api_send_compat():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()

    display = session.get("last_pick_display")
    actual = session.get("last_pick_actual")
    last_text_kw = session.get("last_text_kw")
    last_file_rel = session.get("last_file_rel")

    issue_type, meta = detect_issue(get_conf()["mode"], text, display, actual, last_text_kw, last_file_rel)
    strategy = get_conf()["strategy"]
    hint = actual["rel"] if actual else "无"

    if strategy == "B" and issue_type != "none":
        if issue_type in {"mismatch", "carryover"}:
            modal = {"title": "为确保准确，请选择你要咨询的对象：",
                     "options": ["按文本继续", "按文件继续", "不确定，先帮我识别"]}
        else:
            modal = {"title": "你指的是哪一个？",
                     "options": ["选项1", "选项2", "我不确定，先帮我标注/识别"]}
        log_event("llm_reply", {"modal": modal})
        session["last_text_kw"] = extract_keywords_from_text(text)
        session["last_file_rel"] = actual.get("rel") if actual else None
        return jsonify(ok=True, assistant_html="", modal=modal)

    try:
        r = call_llm(strategy, text, hint, stream=False)
        j = r.json()
        ans = j.get("choices", [{}])[0].get("message", {}).get("content", "(空响应)")
    except Exception:
        ans = f"(本地占位响应) 策略{strategy}：问题={text!r}；线索={hint!r}"

    toast = None
    if strategy == "A" and issue_type in {"mismatch", "carryover"}:
        toast = "检测到可能不一致，已按文件线索自动更正。"

    html = f"<div class='assistant-msg'><p>{ans}</p></div>"
    log_event("llm_reply", {"reply_html": html, "toast": toast})

    session["last_text_kw"] = extract_keywords_from_text(text)
    session["last_file_rel"] = actual.get("rel") if actual else None
    return jsonify(ok=True, assistant_html=html, toast=toast)


# ========= 行为日志 =========
@app.route("/api/log", methods=["POST"])
def api_log():
    data = request.get_json(force=True)
    events = data.get("events") or []
    now = int(time.time() * 1000)
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
