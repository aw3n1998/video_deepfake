# AI 视频智能重绘工作站

通用 AI 视频转视频工具 — 上传视频 + 提示词，AI 重绘视频内容，保留原始结构和动作。

## 核心特性

- **真正的 Video-to-Video**: 使用 img2img 管线，以原始帧为基础进行重绘（而非从零生成）
- **结构保持**: Depth ControlNet 锁定物体结构和空间关系
- **参考图引导**: IP-Adapter 支持，上传参考图引导生成风格
- **自然无 AI 感**: 颜色校正 + 时序平滑 + 亮度去闪烁
- **灵活控制**: strength 参数精确控制原始内容保留比例

## 快速开始

### 1. 安装依赖

```bash
# 先安装 PyTorch (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r requirements.txt
```

### 2. 启动界面

```bash
python app.py
# 打开 http://127.0.0.1:7860
```

### 3. 命令行使用

```bash
# 基本用法
python run_vid2vid.py --input video.mp4 --prompt "cinematic, warm lighting"

# 使用参考图
python run_vid2vid.py --input video.mp4 --prompt "professional photo" --ref-image style.jpg

# 调整重绘强度 (越低越接近原视频)
python run_vid2vid.py --input video.mp4 --prompt "油画风格" --strength 0.5
```

## 参数说明

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `strength` | 0.35 | 0.1~0.9 | 重绘强度，越低越接近原视频 |
| `steps` | 25 | 10~50 | 推理步数，越高质量越好 |
| `resolution` | 768 | 512~1024 | 处理分辨率 |
| `guidance_scale` | 7.5 | 1~20 | CFG 引导系数 |
| `cn_scale` | 0.8 | 0~2 | ControlNet 结构锁定强度 |
| `temporal_smoothing` | 0.12 | 0~0.5 | 时序平滑强度 |

## 技术架构

```
输入视频帧
    ↓
深度估计 (DPT-Large) → 深度图
    ↓
SD 1.5 img2img + Depth ControlNet
  ├─ 原始帧作为 image (保留内容)
  ├─ 深度图作为 control_image (保留结构)
  ├─ 提示词引导风格修改
  └─ IP-Adapter 注入参考图风格 (可选)
    ↓
后处理
  ├─ LAB 颜色校正 (匹配原视频色调)
  ├─ 帧间时序混合 (减少闪烁)
  └─ 亮度去闪烁 (平滑亮度曲线)
    ↓
FFmpeg 合并原始音频 → 输出视频
```

## 项目结构

```
├── app.py              # Gradio 界面
├── run_vid2vid.py      # CLI 入口
├── config.yaml         # 配置文件
├── requirements.txt    # 依赖列表
└── src/
    ├── __init__.py     # 包导出
    ├── vid2vid.py      # 核心 vid2vid 管线
    ├── video_io.py     # 视频 I/O 工具
    └── utils.py        # 通用工具
```
