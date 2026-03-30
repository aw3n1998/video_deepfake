#!/usr/bin/env python3
"""
CLI: 高质量视频重绘 (Vid2Vid)

示例:
  python run_vid2vid.py --input video.mp4 --prompt "cinematic, warm lighting" --output result.mp4
  python run_vid2vid.py --input video.mp4 --prompt "oil painting style" --strength 0.5
  python run_vid2vid.py --input video.mp4 --prompt "anime style" --ref-image style.jpg
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="AI 视频智能重绘 (Vid2Vid)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python run_vid2vid.py --input video.mp4 --prompt "cinematic, warm lighting"

  # 使用参考图引导风格
  python run_vid2vid.py --input video.mp4 --prompt "professional photo" --ref-image ref.jpg

  # 调整重绘强度 (0.2=微调, 0.5=中度, 0.8=大幅重绘)
  python run_vid2vid.py --input video.mp4 --prompt "oil painting" --strength 0.5
        """,
    )
    parser.add_argument("--input", required=True, help="输入视频路径")
    parser.add_argument("--output", default="output_vid2vid.mp4", help="输出视频路径")
    parser.add_argument("--prompt", required=True, help="正向提示词")
    parser.add_argument("--ref-image", default=None, help="参考图路径 (可选)")
    parser.add_argument(
        "--neg-prompt",
        default="low quality, blurry, deformed, ugly, bad anatomy, watermark",
        help="负向提示词",
    )
    parser.add_argument(
        "--steps", type=int, default=25, help="推理步数 (默认 25, 推荐 20~35)"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.35,
        help="重绘强度 (默认 0.35, 越低越接近原视频)",
    )
    parser.add_argument(
        "--resolution", type=int, default=768, help="处理分辨率 (默认 768)"
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=7.5, help="CFG 引导系数"
    )
    parser.add_argument(
        "--cn-scale",
        type=float,
        default=0.8,
        help="ControlNet 结构锁定强度 (默认 0.8)",
    )
    parser.add_argument(
        "--temporal-smoothing",
        type=float,
        default=0.12,
        help="时序平滑强度 (默认 0.12, 0=关闭)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        sys.exit(1)

    try:
        from src.vid2vid import Vid2VidPipeline

        logger.info("初始化 Vid2Vid 管线...")
        pipeline = Vid2VidPipeline(device="cuda")

        logger.info(f"视频: {args.input}")
        logger.info(f"提示词: {args.prompt}")
        logger.info(f"强度: {args.strength}")
        if args.ref_image:
            logger.info(f"参考图: {args.ref_image}")

        success = pipeline.process_video(
            video_path=args.input,
            output_path=args.output,
            prompt=args.prompt,
            negative_prompt=args.neg_prompt,
            reference_image=args.ref_image,
            num_steps=args.steps,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.cn_scale,
            resolution=args.resolution,
            temporal_smoothing=args.temporal_smoothing,
        )

        if success:
            logger.info(f"✅ 完成！输出: {args.output}")
        else:
            logger.error("❌ 处理失败")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
