# -*- coding: utf-8 -*-

import os
from pathlib import Path

# === 运行参数 ===
HOST = "0.0.0.0"
PORT = 5010
DEBUG = True

# === 模型后端（OpenAI 兼容）===
# 你可以用环境变量覆盖这些值；若不设环境变量就用这里的默认。
AIECNU_API_KEY = os.environ.get(
    "AIECNU_API_KEY",
    "sk-xtkMxpjDtHcQc29t1wvbHzOk8OijFiNAAqs2dAtLxq1ThFc3",
)
AIECNU_BASE_URL = os.environ.get(
    "AIECNU_BASE_URL",
    "http://49.51.37.239:3006/v1",
)
MODEL = os.environ.get("MODEL", "gpt-4.1")

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
