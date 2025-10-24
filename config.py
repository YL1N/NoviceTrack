# -*- coding: utf-8 -*-

import os
from pathlib import Path

# === 运行参数 ===
HOST = "0.0.0.0"
PORT = 5010
DEBUG = True

# === 模型后端（OpenAI 兼容 /chat/completions）===
# 可通过环境变量覆盖：
#   MODELSCOPE_API_KEY      (例如 ms-xxxxxxxx... )
#   MODELSCOPE_BASE_URL     默认 https://api-inference.modelscope.cn/v1
#   MODEL                   默认 Qwen/Qwen3-VL-235B-A22B-Instruct
MODELSCOPE_API_KEY = os.environ.get("MODELSCOPE_API_KEY", "ms-4c5af8e1-b8d6-4abc-90cc-d4fb078702bb")
MODELSCOPE_BASE_URL = os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")

# === 资源与日志路径 ===
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets" / "candidates"   # ← 候选文件根目录
LOG_DIR = BASE_DIR / "data" / "logs"              # ← 行为日志目录
LOG_DIR.mkdir(parents=True, exist_ok=True)

# === 选择器显示设置 ===
GRID_COLS = 4              # 网格列数（用于“邻居”欺骗）
SHOW_HIDDEN = False        # 是否展示以点开头的隐藏文件

# === 允许展示的文件类型（前端能预览图片；其他类型以文件卡展示）===
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
