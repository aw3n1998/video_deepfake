"""
Google Colab 一键安装 + 运行脚本
把下面的代码块依次粘贴到 Colab 的代码单元格中执行

环境要求: Colab + GPU (T4 或更好)
"""

# ===========================================================
# 【代码块 1】安装依赖（约5~10分钟，只需运行一次）
# ===========================================================
INSTALL = """
# 升级 pip
pip install -q --upgrade pip

# PyTorch (Colab 已自带，确认版本)
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| 版本:', torch.__version__)"

# InsightFace + ONNX
pip install -q insightface==0.7.3 onnxruntime-gpu

# Stable Diffusion + ControlNet
pip install -q diffusers==0.27.2 transformers accelerate xformers

# ControlNet 辅助工具（OpenPose 姿态提取）
pip install -q controlnet-aux

# MediaPipe 人体分割
pip install -q mediapipe

# 语音处理
pip install -q faster-whisper edge-tts

# 视频处理
pip install -q opencv-python-headless moviepy ffmpeg-python pydub librosa soundfile

# 其他工具
pip install -q pyyaml tqdm

echo "=== 安装完成 ==="
"""

# ===========================================================
# 【代码块 2】克隆项目 + 下载模型
# ===========================================================
CLONE_AND_MODELS = """
import os

# 克隆项目（已克隆则跳过）
if not os.path.exists('video_deepfake'):
    !git clone https://github.com/aw3n1998/video_deepfake.git

os.chdir('video_deepfake')
!ls

# 创建模型目录
os.makedirs('models', exist_ok=True)

# 下载 InsightFace 换脸模型（约500MB）
if not os.path.exists('models/inswapper_128.onnx'):
    print("下载 inswapper_128.onnx ...")
    !wget -q --show-progress -O models/inswapper_128.onnx \\
        "https://hf-mirror.com/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    print("下载完成")
else:
    print("inswapper_128.onnx 已存在，跳过")

# InsightFace buffalo_l 模型（首次运行自动下载到 ~/.insightface）
print("预热 InsightFace 模型...")
import insightface
app = insightface.app.FaceAnalysis(
    name='buffalo_l',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(640, 640))
print("InsightFace 准备完成")
"""

# ===========================================================
# 【代码块 3】上传视频和人脸图片
# ===========================================================
UPLOAD = """
from google.colab import files
import shutil

print("=== 上传输入视频 ===")
uploaded = files.upload()
input_video = list(uploaded.keys())[0]
print(f"视频已上传: {input_video}")

print("\\n=== 上传女性人脸图片 ===")
uploaded2 = files.upload()
face_img = list(uploaded2.keys())[0]
print(f"人脸图片已上传: {face_img}")
"""

# ===========================================================
# 【代码块 4】运行完整转换（全身重绘 + 换脸 + 换声）
# ===========================================================
RUN = """
import sys
sys.path.insert(0, '.')

# 使用国内镜像
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from run_full_transform import process

success = process(
    input_video=input_video,   # 上一步上传的视频
    female_face=face_img,      # 上一步上传的人脸图
    output_path='output_full.mp4',
    tts_voice='zh-CN-XiaoxiaoNeural',  # 女声音色
    asr_lang='zh',             # 视频语言
    skip_frames=2,             # 跳帧=2: 每3帧重绘1帧（速度快3倍）
    strength=0.75,             # 重绘强度 0.6~0.85
)

if success:
    print("\\n[OK] 处理完成！")
else:
    print("\\n[FAIL] 处理失败，请查看上方日志")
"""

# ===========================================================
# 【代码块 5】下载结果
# ===========================================================
DOWNLOAD = """
from google.colab import files
import os

if os.path.exists('output_full.mp4'):
    size = os.path.getsize('output_full.mp4') / 1024 / 1024
    print(f"输出文件大小: {size:.1f} MB")
    files.download('output_full.mp4')
else:
    print("输出文件不存在，请检查处理日志")
"""

# ===========================================================
# 打印使用说明
# ===========================================================
if __name__ == '__main__':
    print("""
=== Colab 使用步骤 ===

1. 打开 Google Colab，确认右上角已选择 GPU（T4）

2. 依次创建5个代码单元格，粘贴对应内容:

   单元格1: 安装依赖
   单元格2: 克隆项目 + 下载模型
   单元格3: 上传视频和人脸图片
   单元格4: 运行转换
   单元格5: 下载结果

3. 关键参数说明:
   skip_frames = 0  # 最好质量，最慢
   skip_frames = 2  # 推荐，速度快3倍
   strength = 0.75  # 重绘强度，可调 0.6~0.85

4. T4 GPU 处理速度参考（125秒视频）:
   skip_frames=2: 约 20~40 分钟
   skip_frames=0: 约 60~90 分钟
""")
