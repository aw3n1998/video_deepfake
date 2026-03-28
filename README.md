# 视频深度伪造处理器 (Video Deepfake Processor)

一个完整的、可扩展的视频深度伪造处理框架，支持 **人脸交换、TTS 语音合成、音视频合成** 等功能。

## 🎯 核心特性

- ✅ **多种人脸检测引擎** (MTCNN、MediaPipe、YOLOv8)
- ✅ **高质量人脸交换** (InsightFace / Deepfaceslive)
- ✅ **免费 TTS 引擎** (Microsoft Edge TTS)
- ✅ **完整的音视频处理** (FFmpeg 集成)
- ✅ **模块化设计** (可独立使用各模块)
- ✅ **批量处理支持**
- ✅ **完整的日志和错误处理**

## 📋 系统要求

### 操作系统
- Linux (推荐: Ubuntu 18.04+)
- macOS 10.14+
- Windows 10+ (需要安装 FFmpeg)

### 硬件要求
- CPU: Intel i5 或更好
- 内存: 8GB 最小，16GB 推荐
- GPU: 可选 (NVIDIA CUDA 支持更快处理)
- 磁盘: 至少 20GB 空闲空间

### 软件依赖
- Python 3.8+
- FFmpeg 4.4+
- Git

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/your-repo/video-deepfake-processor.git
cd video-deepfake-processor

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 安装 FFmpeg

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
下载: https://ffmpeg.org/download.html
或使用 Chocolatey:
```bash
choco install ffmpeg
```

### 3. 下载预训练模型

```bash
# 创建模型目录
mkdir -p models

# InsightFace 模型会在首次使用时自动下载
# 或手动下载: https://github.com/deepinsight/insightface
```

### 4. 完整工作流程示例

```bash
python -m src.pipeline source_face.jpg input_video.mp4 "你好，这是合成音频" output.mp4
```

## 📖 详细使用指南

### 方式 1: 使用完整工作流 (推荐)

```python
from src.pipeline import VideoDeepfakePipeline

# 初始化工作流
pipeline = VideoDeepfakePipeline()

# 执行完整处理
success = pipeline.process_full_pipeline(
    source_image='source_face.jpg',
    input_video='input_video.mp4',
    output_text='你好，世界',
    output_path='output.mp4'
)

if success:
    print("✓ 处理完成！")
```

### 方式 2: 分步使用各模块

#### 人脸检测

```python
from src.face_detector import FaceDetector
import cv2

# 初始化检测器
detector = FaceDetector(model_type='mtcnn')

# 加载图片
image = cv2.imread('photo.jpg')

# 检测人脸
faces = detector.detect(image, confidence_threshold=0.9)

print(f"检测到 {len(faces)} 张人脸")
for face in faces:
    print(f"  位置: {face['bbox']}, 置信度: {face['confidence']:.2f}")

# 绘制检测结果
result = detector.draw_faces(image, faces)
cv2.imwrite('result.jpg', result)
```

#### 人脸交换

```python
from src.face_swapper import FaceSwapper
import cv2

swapper = FaceSwapper()

source = cv2.imread('source_face.jpg')
target = cv2.imread('target_image.jpg')

# 交换人脸
result = swapper.swap_faces(source, target)

cv2.imwrite('swapped.jpg', result)

# 视频人脸交换
swapper.swap_faces_in_video(
    source_image_path='source_face.jpg',
    video_path='input.mp4',
    output_path='swapped_video.mp4'
)
```

#### TTS 语音合成

```python
from src.audio_processor import TTSProcessor

tts = TTSProcessor(tts_engine='edge-tts')

# 文字转语音
success = tts.text_to_speech(
    text='你好，这是合成的语音',
    output_path='audio.mp3',
    voice='zh-CN-XiaoxiaoNeural'
)

# 批量处理
texts = ['你好', '世界', '再见']
audio_files = tts.batch_text_to_speech(
    texts=texts,
    output_dir='./audio_output',
    voice='zh-CN-XiaoxiaoNeural'
)
```

#### 视频合成

```python
from src.video_composer import VideoComposer

composer = VideoComposer()

# 添加音频到视频
success = composer.compose_video_with_audio(
    video_path='video.mp4',
    audio_path='audio.mp3',
    output_path='final.mp4'
)

# 合并多个视频
videos = ['video1.mp4', 'video2.mp4', 'video3.mp4']
composer.merge_videos(videos, 'merged.mp4')

# 添加水印
composer.add_watermark(
    video_path='video.mp4',
    watermark_path='watermark.png',
    output_path='watermarked.mp4',
    position='bottom-right',
    scale=0.1
)

# 调整分辨率
composer.resize_video(
    video_path='video.mp4',
    output_path='resized.mp4',
    width=1280,
    height=720
)
```

### 批量处理

```python
from src.pipeline import VideoDeepfakePipeline

pipeline = VideoDeepfakePipeline()

# 定义批处理任务
tasks = [
    {
        'source_image': 'face1.jpg',
        'input_video': 'video1.mp4',
        'output_text': '第一个视频',
        'output_path': 'output1.mp4'
    },
    {
        'source_image': 'face2.jpg',
        'input_video': 'video2.mp4',
        'output_text': '第二个视频',
        'output_path': 'output2.mp4'
    },
]

# 执行批处理
results = pipeline.batch_process(tasks)

print(f"成功: {results['success']}, 失败: {results['failed']}")
```

## 🔧 配置说明

编辑 `config.yaml` 文件来自定义各模块的参数：

```yaml
# 人脸检测
face_detector:
  model_type: mtcnn              # mtcnn | mediapipe | yolov8
  confidence_threshold: 0.9

# 人脸交换
face_swapper:
  model_name: inswapper_128.onnx

# TTS 配置
tts:
  engine: edge-tts               # edge-tts | pyttsx3 | xunfei
  voice: zh-CN-XiaoxiaoNeural   # 语音选择
  rate: '+0%'                    # 语速调整

# 视频输出
video:
  codec: h264                    # h264 | h265 | vp9
  bitrate: 5000k
```

## 📊 性能对比

| 模块 | 模型 | 速度 | 质量 | 内存 |
|------|------|------|------|------|
| 人脸检测 | MTCNN | 中等 | 高 | 中等 |
| 人脸检测 | MediaPipe | 快 | 中等 | 低 |
| 人脸检测 | YOLOv8 | 快 | 高 | 低 |
| 人脸交换 | InsightFace | 中等 | 高 | 高 |
| TTS | Edge TTS | 快 | 高 | 低 |
| TTS | pyttsx3 | 快 | 低 | 低 |

## 🔐 注意事项

⚠️ **重要法律声明**

本工具仅供学习和研究使用。使用本工具进行的以下行为可能违反法律：
- 未经同意创建虚假视频
- 用于欺诈、骚扰或损害他人声誉
- 创建非法内容

用户必须遵守当地法律和道德规范。开发者对任何不当使用不负责任。

## 📝 日志

所有操作都会记录到 `logs/` 目录：
- `pipeline.log` - 完整工作流日志
- `batch_result_*.json` - 批处理结果

查看日志：
```bash
tail -f logs/pipeline.log
```

## 🐛 故障排除

### 问题: ModuleNotFoundError: No module named 'insightface'

**解决方案:**
```bash
pip install insightface
```

### 问题: FFmpeg not found

**解决方案:**
确保 FFmpeg 已安装并在 PATH 中。

### 问题: CUDA out of memory

**解决方案:**
1. 使用 CPU 模式（更慢但更稳定）
2. 减少视频分辨率
3. 增加系统内存

### 问题: TTS 超时

**解决方案:**
1. 检查网络连接（Edge TTS 需要联网）
2. 使用本地 TTS (pyttsx3)
3. 增加超时时间

## 🤝 贡献

欢迎提交 Pull Request 或报告 Issue！

## 📄 许可证

MIT License

## 🙏 致谢

感谢以下开源项目的贡献：
- [InsightFace](https://github.com/deepinsight/insightface)
- [DeepFaceLive](https://github.com/iperov/DeepFaceLive)
- [OpenCV](https://opencv.org/)
- [FFmpeg](https://ffmpeg.org/)

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送 Email

## 🎓 学习资源

- [人脸识别概述](https://en.wikipedia.org/wiki/Facial_recognition_system)
- [深度学习 OpenCV](https://docs.opencv.org/)
- [FFmpeg 官方文档](https://ffmpeg.org/documentation.html)

---

**最后更新**: 2024-03-24
**版本**: 1.0.0
