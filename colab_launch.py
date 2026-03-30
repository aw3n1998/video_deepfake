#!/usr/bin/env python3
"""
Google Colab 一键部署脚本

使用方法：在 Colab 新建笔记本，粘贴以下代码到一个单元格运行：

    !git clone https://github.com/aw3n1998/video_deepfake.git
    %cd video_deepfake
    !python colab_launch.py
"""

import subprocess
import sys
import os


def run(cmd, check=True):
    print(f"\n{'='*60}")
    print(f"▶ {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode == 0


def main():
    # ──────────────────────────────────────────
    # 1. 检测环境
    # ──────────────────────────────────────────
    print("\n🔍 检测 Colab 环境...")

    # 检查 GPU
    gpu_ok = run("nvidia-smi", check=False)
    if not gpu_ok:
        print("⚠️ 未检测到 GPU，部分功能可能无法使用")

    # 检查 PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError:
        print("📦 安装 PyTorch...")
        run("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

    # ──────────────────────────────────────────
    # 2. 安装依赖
    # ──────────────────────────────────────────
    print("\n📦 安装项目依赖...")
    # 强制重新安装 matplotlib 以匹配当前 numpy 版本，防止 numpy 更新后 C 扩展报错
    run("pip install -q -r requirements.txt")
    run("pip install -q -U --force-reinstall matplotlib")

    # FFmpeg (Colab 通常已预装)
    run("ffmpeg -version > /dev/null 2>&1 || apt-get install -y ffmpeg", check=False)

    # ──────────────────────────────────────────
    # 3. 设置环境变量
    # ──────────────────────────────────────────
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"

    # ──────────────────────────────────────────
    # 4. 启动 Gradio (share=True 生成公网链接)
    # ──────────────────────────────────────────
    print("\n" + "="*60)
    print("🚀 启动 AI 视频智能重绘工作站...")
    print("   模型将在首次使用时自动下载 (约 5-8GB)")
    print("   启动后会生成一个 gradio.live 公网链接")
    print("="*60 + "\n")

    # 直接导入并启动，这样日志能实时显示
    sys.path.insert(0, ".")
    from app import build_ui

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,         # 生成公网链接
        quiet=False,
    )


if __name__ == "__main__":
    main()
