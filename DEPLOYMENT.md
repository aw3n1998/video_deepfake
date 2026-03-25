# 部署和集成指南

## 📦 完整安装步骤

### 第一步: 系统依赖安装

#### Linux (Ubuntu 20.04+)

```bash
# 更新包管理器
sudo apt-get update
sudo apt-get upgrade -y

# 安装系统依赖
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev

# 安装 CUDA（可选，用于 GPU 加速）
# sudo apt-get install -y nvidia-driver-XXX nvidia-cuda-toolkit
```

#### macOS

```bash
# 使用 Homebrew
brew install python@3.10
brew install ffmpeg
brew install git
```

#### Windows

```bash
# 使用 Chocolatey
choco install python ffmpeg git

# 或手动下载:
# Python: https://www.python.org/downloads/
# FFmpeg: https://ffmpeg.org/download.html
# Git: https://git-scm.com/download/win
```

### 第二步: 项目设置

```bash
# 克隆项目
git clone https://github.com/your-repo/video-deepfake-processor.git
cd video-deepfake-processor

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 升级 pip
pip install --upgrade pip setuptools wheel

# 安装依赖
pip install -r requirements.txt

# 创建必要的目录
mkdir -p models logs data/{input,output,cache}
```

### 第三步: 下载预训练模型

```bash
# 模型会在首次使用时自动下载，但也可以手动下载:

# 创建模型目录
mkdir -p models/face_detection
mkdir -p models/deepfaceslive

# InsightFace 模型会自动下载到:
# ~/.insightface/models/

# 如需离线模型，请参考:
# https://github.com/deepinsight/insightface/wiki
```

### 第四步: 配置文件

编辑 `config.yaml`:

```yaml
work_dir: ./data
face_detector:
  model_type: mtcnn
  confidence_threshold: 0.9
tts:
  engine: edge-tts
  voice: zh-CN-XiaoxiaoNeural
video:
  codec: h264
  bitrate: 5000k
```

### 第五步: 测试安装

```bash
# 运行快速测试
python quickstart.py

# 或运行单元测试
pytest tests/ -v
```

## 🚀 与 Spring Boot 集成

### 方案 1: 微服务架构（推荐）

#### 第一步: 创建 Python Flask API

```python
# api/app.py
from flask import Flask, request, jsonify
from src.pipeline import VideoDeepfakePipeline
import os
import json

app = Flask(__name__)
pipeline = VideoDeepfakePipeline()

@app.route('/api/process', methods=['POST'])
def process_video():
    """处理视频的 API 端点"""
    try:
        data = request.json
        
        success = pipeline.process_full_pipeline(
            source_image=data['source_image'],
            input_video=data['input_video'],
            output_text=data['output_text'],
            output_path=data.get('output_path', 'output.mp4')
        )
        
        if success:
            return jsonify({'status': 'success', 'message': 'Processing completed'})
        else:
            return jsonify({'status': 'failed', 'message': 'Processing failed'}), 400
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/detect-faces', methods=['POST'])
def detect_faces():
    """人脸检测 API"""
    try:
        image_path = request.json['image_path']
        from src.face_detector import quick_detect
        
        faces = quick_detect(image_path)
        return jsonify({
            'status': 'success',
            'face_count': len(faces),
            'faces': faces
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """TTS API"""
    try:
        text = request.json['text']
        voice = request.json.get('voice', 'zh-CN-XiaoxiaoNeural')
        output_path = f"audio_{int(time.time())}.mp3"
        
        from src.audio_processor import quick_tts
        success = quick_tts(text, output_path, voice)
        
        if success:
            return jsonify({'status': 'success', 'audio_path': output_path})
        else:
            return jsonify({'status': 'failed'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

#### 第二步: 在 Spring Boot 中调用 Python API

```java
// PythonServiceClient.java
@Service
public class PythonServiceClient {
    
    @Value("${python.api.url:http://localhost:5000}")
    private String pythonApiUrl;
    
    private final RestTemplate restTemplate;
    
    public String processVideo(String sourceImage, String inputVideo, String outputText) {
        Map<String, String> request = new HashMap<>();
        request.put("source_image", sourceImage);
        request.put("input_video", inputVideo);
        request.put("output_text", outputText);
        
        try {
            String url = pythonApiUrl + "/api/process";
            ResponseEntity<Map> response = restTemplate.postForEntity(url, request, Map.class);
            
            if (response.getStatusCode().is2xxSuccessful()) {
                return "success";
            }
        } catch (Exception e) {
            logger.error("Python API call failed", e);
        }
        
        return "failed";
    }
}
```

#### 第三步: Docker 部署

创建 `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 5000

# 启动服务
CMD ["python", "api/app.py"]
```

创建 `docker-compose.yml`:

```yaml
version: '3.8'

services:
  python-processor:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
    networks:
      - processing-network

  spring-boot-app:
    image: spring-boot-app:latest
    ports:
      - "8080:8080"
    depends_on:
      - python-processor
    environment:
      - PYTHON_API_URL=http://python-processor:5000
    networks:
      - processing-network

networks:
  processing-network:
    driver: bridge
```

启动服务:

```bash
docker-compose up -d
```

### 方案 2: 直接调用 Python 脚本

```java
// DirectPythonCaller.java
public String processPythonPipeline(VideoProcessRequest request) throws Exception {
    ProcessBuilder pb = new ProcessBuilder(
        "/path/to/venv/bin/python",
        "-m", "src.pipeline",
        request.getSourceImage(),
        request.getInputVideo(),
        request.getOutputText(),
        request.getOutputPath()
    );
    
    pb.directory(new File("/path/to/project"));
    Process process = pb.start();
    
    // 等待完成
    int exitCode = process.waitFor();
    
    if (exitCode == 0) {
        return request.getOutputPath();
    } else {
        throw new RuntimeException("Python processing failed");
    }
}
```

## 📊 性能优化

### 1. GPU 加速

```python
# 配置 GPU
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU")
```

配置文件:

```yaml
# config.yaml
face_swapper:
  use_gpu: true
  device: cuda:0
```

### 2. 多进程处理

```python
# pipeline.py
from multiprocessing import Pool

def process_batch_parallel(tasks, num_processes=4):
    """并行处理多个任务"""
    with Pool(num_processes) as pool:
        results = pool.map(process_single_task, tasks)
    return results
```

### 3. 缓存优化

```python
# 使用 Redis 缓存中间结果
import redis

cache = redis.Redis(host='localhost', port=6379, db=0)

def get_or_process_faces(image_path):
    """缓存人脸检测结果"""
    cache_key = f"faces:{hash(image_path)}"
    
    cached = cache.get(cache_key)
    if cached:
        return json.loads(cached)
    
    faces = detector.detect(cv2.imread(image_path))
    cache.setex(cache_key, 3600, json.dumps(faces))  # 缓存 1 小时
    
    return faces
```

## 📈 监控和日志

### 日志配置

```python
# logging.conf
[loggers]
keys=root,processor

[handlers]
keys=console,file

[formatters]
keys=standard

[logger_root]
level=DEBUG
handlers=console,file

[logger_processor]
level=INFO
handlers=console,file
qualname=src.pipeline
propagate=0

[handler_console]
class=StreamHandler
level=INFO
formatter=standard
args=(sys.stdout,)

[handler_file]
class=FileHandler
level=DEBUG
formatter=standard
args=('logs/pipeline.log', 'a')

[formatter_standard]
format=%(asctime)s [%(levelname)s] %(name)s: %(message)s
```

### Prometheus 监控

```python
# metrics.py
from prometheus_client import Counter, Histogram, start_http_server

# 定义指标
processing_total = Counter(
    'video_processing_total',
    'Total number of videos processed',
    ['status']
)

processing_duration = Histogram(
    'video_processing_duration_seconds',
    'Video processing duration'
)

# 在代码中使用
@processing_duration.time()
def process_video(source, target, text):
    try:
        result = pipeline.process_full_pipeline(source, target, text, 'output.mp4')
        processing_total.labels(status='success').inc()
        return result
    except Exception as e:
        processing_total.labels(status='failed').inc()
        raise

# 启动 Prometheus 服务
if __name__ == '__main__':
    start_http_server(8000)  # 暴露 :8000/metrics
```

## 🔍 故障排除

### 常见问题

| 问题 | 解决方案 |
|------|--------|
| CUDA 内存不足 | 1. 使用 CPU 模式<br>2. 降低视频分辨率<br>3. 增加 GPU 内存 |
| FFmpeg 未找到 | 确保 FFmpeg 在 PATH 中 |
| InsightFace 模型下载失败 | 检查网络，手动下载模型 |
| TTS 超时 | 检查网络连接或使用本地 TTS |

### 日志分析

```bash
# 查看实时日志
tail -f logs/pipeline.log

# 搜索错误
grep ERROR logs/pipeline.log

# 统计处理时间
grep "完成" logs/pipeline.log | grep -o "处理时间: [0-9]*" | awk '{s+=$3} END {print "平均时间:", s/NR}'
```

## 🚢 生产部署清单

- [ ] 所有依赖已安装
- [ ] FFmpeg 已配置
- [ ] 模型已下载
- [ ] 配置文件已修改
- [ ] 日志目录已创建
- [ ] 数据目录有足够空间
- [ ] 防火墙规则已配置
- [ ] API 密钥已配置
- [ ] 监控已设置
- [ ] 备份策略已制定
- [ ] 负载测试已通过
- [ ] 文档已更新

## 📞 支持

遇到问题？
1. 查看日志文件
2. 运行 `pytest tests/ -v` 测试各模块
3. 提交 Issue（包含完整错误日志）

---

**更新时间**: 2024-03-24
