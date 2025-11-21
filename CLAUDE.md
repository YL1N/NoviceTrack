# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NoviceTrack 是一个基于 Flask 的多模态 AI 实验平台，使用 Qwen3-VL 视觉语言模型进行图文冲突仲裁研究。系统研究 AI 助手如何处理用户文本描述与上传图片之间的冲突。

## Quick Start

```bash
# 安装依赖
pip install -r requirements.txt

# 运行应用
python app.py
```

服务器启动在 `http://0.0.0.0:5010`（可通过环境变量配置）。

## 环境配置

关键配置通过环境变量管理（见 config.py）：

- `DASHSCOPE_API_KEY`: DashScope/ModelScope API 密钥（Qwen 模型访问）
- `MODEL`: 模型名称（默认："qwen3-vl-plus"）
- `HOST`, `PORT`: 服务器绑定（默认：0.0.0.0:5010）
- `DEBUG`: 调试模式（默认：1）
- `ASSETS_DIR`: 资源目录（默认：`assets/candidates/`）
- `LOG_DIR`: 日志目录（默认：`data/logs/`）
- `THUMB_DIR`: 缩略图目录（默认：`assets/_thumbs/`）

## 常用开发命令

```bash
# 启动开发服务器
python app.py

# 测试 API 连通性
curl http://localhost:5010/api/upstream_ping

# 查看日志
ls data/logs/

# 清理缩略图缓存
rm -rf assets/_thumbs/*
```

## 依赖说明

核心依赖（requirements.txt）：
- **Flask 3.0.3**: Web 框架
- **requests 2.32.3**: HTTP 客户端（API 调用）
- **Pillow 10.4.0**: 可选的图像处理（用于缩略图生成）
- **python-dotenv 1.0.1**: 可选的 .env 文件支持
- **gunicorn 22.0.0**: 可选的部署 WSGI 服务器

Pillow 是可选的，但如果缺失，某些图片可能无法生成缩略图。

## 核心架构

### 图文仲裁流程

1. **用户输入** → 文本 + 图片附件
2. **图像标注** (`image_brief`)：调用视觉模型分类图片（车辆/手机/草本/风景/水果/电器/试卷/药品）
3. **冲突检测** (`arbiter_decide`)：对比用户文本与图片分类，识别冲突
4. **路由决策**：
   - `NORMAL`: 无冲突，使用中性提示词
   - `CONFLICT`: 检测到冲突，启用专门系统提示词
   - `TEXT_ONLY`/`IMAGE_ONLY`: 用户明确确认以文本或图片为准
5. **生成响应**：使用 Qwen3-VL 生成最终回复

### 会话管理

- **session_id**: 每个浏览器会话唯一标识
- **trial_id**: 每个对话 trial 独立追踪（区别于 session）
- **模式切换**：切换模式（free/task_i/task_ii/task_iii）会创建新 trial 并清空对话
- **历史追踪**：会话历史存储在 `_GLOBAL_CHAT_HISTORY`（app.py:169-210）

### 冲突检测逻辑

仲裁器（app.py:712-848）在以下情况检测冲突：
- **品牌/型号不匹配**：用户说"特斯拉"但图片显示比亚迪（置信度≥0.6）
- **类别不匹配**：文本说"苹果"但图片显示橙子
- **属性冲突**：文本说"红苹果"但图片显示青苹果
- **多实例指代不清**：文本用单数（"这个药盒"）但图片显示多个相似对象

## Task 模式说明

**Task I**（app.py:1082-1094）：
- "邻居欺骗"模式
- 选择的图片被替换为同一目录或相邻索引的随机图片
- 用于实验研究用户对图片替换的感知

**Task II/III**：保留用于未来实验条件

## 调试技巧

设置 `DEBUG_MODE = 1`（app.py:39）启用详细调试输出：
- 上游 API 请求/响应（app.py:1018-1025）
- 仲裁决策信息
- 限流状态
- 图片选择列表

## API 端点速查

- `GET /`: UI 主页面
- `GET /api/picker_list`: 列出可用资源
- `POST /api/pick`: 用户选择资源（处理 Task I 欺骗）
- `POST /api/remove_pick`: 移除已选资源
- `GET /thumb/<path>`: 惰性缩略图生成器
- `POST /api/send_stream`: 主要推理端点（SSE 流式）
- `POST /api/send`: 备用推理端点（非流式）
- `POST /api/set_line`: 切换策略（A/B/C）
- `POST /api/set_mode`: 切换实验模式（free/I/II/III）
- `POST /api/new_chat`: 重置会话（保持 session）
- `POST /api/log`: 客户端日志记录

## 前端要点

前端（static/app.js）实现：
- **双击选择**：文件通过双击网格单元格选择
- **客户端去重**：防止重复附件
- **SSE 支持**：实时流式响应
- **中断支持**：使用 AbortController

### 已知前端问题

该项目曾存在附件选择器 Bug，详见 [TEST_CASES.md](TEST_CASES.md)：
1. 双击无反应（事件绑定丢失）
2. 模态框不显示图片（渲染时序问题）
3. 新对话后问题重现（状态重置问题）

如需修复，请参考 TEST_CASES.md 中的测试用例。

## 图片处理

- **缩略图**：使用 Pillow 自动生成（downscale_to_b64）
- **内联限制**：350KB（超过会压缩）
- **支持格式**：.png, .jpg, .jpeg, .webp, .gif, .bmp
- **回退机制**：缩略图生成失败时返回原文件

## 限流机制

- **全局限流窗口**：遇到 429 错误时设置 `_RATE_LIMIT_UNTIL`
- **跳过图片标注**：限流窗口期间跳过图像分析以减少 API 调用
- **指数退避**：3 次重试，指数延迟（app.py:408-410）

## 领域特定处理

- **车辆/手机**：规范化品牌/型号别名（如"汉"→"Han"）
- **草本**：区分枸杞/决明子
- **地标**：规范化富士山、西湖
- **水果/电器/试卷/药品**：领域感知的属性提取（颜色、数量、旋钮、面板文字等）

## 重要注意事项

1. **绝不能删除 trial_id 追踪**：对实验设计至关重要（Task I 欺骗）
2. **保持历史管理功能**：用于多轮对话但不重复发送图片
3. **保留空格处理**：资源路径必须 URL 编码（`encodeURIComponent`）
4. **附件生命周期**：上下文切换时清空（模式变更）和新对话
5. **冲突逻辑顺序**：别名归一化 → 置信度阈值 → 默认行为
6. **Pillow 是可选的**：但建议安装以获得更好的缩略图支持

## 测试工具

项目中包含测试文档（TEST_CASES.md），涵盖：
- 附件选择器功能测试
- 新对话场景测试
- 模式切换测试
- 极端场景测试

建议使用 Playwright 或 Selenium 进行自动化 UI 测试。

## 日志分析

日志存储在 `data/logs/`，每个会话一个 JSON 文件：
- 包含用户发送、LLM 响应、仲裁决策等事件
- 可用于实验数据分析和 bug 诊断

## 常见问题

**Q: 图片不显示？**
- 检查 ASSETS_DIR 路径配置
- 确认 Pillow 已安装
- 查看 THUMB_DIR 目录权限

**Q: API 调用失败？**
- 检查 DASHSCOPE_API_KEY 环境变量
- 查看调试输出中的 429 限流信息
- 确认网络连接

**Q: 冲突检测不生效？**
- 设置 DEBUG_MODE = 1 查看仲裁过程
- 检查限流窗口是否跳过标注
- 确认图片标注返回有效结果

所有信息对维护本实验平台的有效性至关重要。