# -*- coding: utf-8 -*-
"""
简易配置（可选）。本项目主要从环境变量读取，如果你不想改环境变量，
把这里的值 export 成环境变量即可：
- MODELSCOPE_API_KEY
- MODELSCOPE_BASE_URL
- MODEL
- HOST / PORT / DEBUG
- ASSETS_DIR / LOG_DIR
"""

import os
from pathlib import Path

# === 运行参数 ===
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5010"))
DEBUG = os.environ.get("DEBUG", "1") == "1"

# === 模型后端（OpenAI 兼容 /chat/completions）===
MODELSCOPE_API_KEY = os.environ.get("MODELSCOPE_API_KEY", "ms-xxxxxxxxxxxxxxxxxxxxxxxxxxxx")
MODELSCOPE_BASE_URL = os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")

# === 资源与日志路径 ===
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = Path(os.environ.get("ASSETS_DIR", str(BASE_DIR / "assets" / "candidates")))
LOG_DIR = Path(os.environ.get("LOG_DIR", str(BASE_DIR / "data" / "logs")))
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
