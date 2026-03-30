#!/usr/bin/env python3
"""
CLI 脚本: 运行高质量视频重绘 (Vid2Vid) 
支持 RTX 5090，可调节 Prompt、分辨率、渲染步数等。
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='通用高质量视频重绘 (Vid2Vid)')
    parser.add_argument('--input', required=True, help="输入视频路径")
    parser.add_argument('--output', default='output_vid2vid.mp4', help="输出视频路径")
    parser.add_argument('--reference-image', default=None, help="参考图片路径 (引导人物或画质)")
    parser.add_argument('--prompt', required=True, help="正向提示词")
    parser.add_argument('--neg-prompt', default="low quality, low res, blurry, ugly, bad anatomy, deformed, signature, watermark", help="负向提示词")
    parser.add_argument('--steps', type=int, default=30, help="渲染步数 (推荐 30~50 获得最高画质)")
    parser.add_argument('--strength', type=float, default=0.75, help="重绘强度 (0~1)")
    parser.add_argument('--resolution', type=int, default=768, help="最大处理分辨率 (推荐 768 或 1024)")
    parser.add_argument('--guidance-scale', type=float, default=7.5, help="CFG Scale")
    parser.add_argument('--pose-weight', type=float, default=0.85, help="OpenPose 权重 (0~1)")
    parser.add_argument('--depth-weight', type=float, default=0.75, help="Depth 权重 (0~1)")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    try:
        from src.vid2vid import Vid2VidPipeline
        
        logger.info("Initializing Vid2VidPipeline...")
        pipeline = Vid2VidPipeline(device="cuda")
        
        logger.info(f"Processing video: {args.input}")
        logger.info(f"Prompt: {args.prompt}")
        logger.info(f"Reference Image: {args.reference_image}")
        
        success = pipeline.process_video(
            video_path=args.input,
            output_path=args.output,
            prompt=args.prompt,
            negative_prompt=args.neg_prompt,
            reference_image=args.reference_image,
            num_steps=args.steps,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=[args.pose_weight, args.depth_weight],
            resolution=args.resolution
        )
        
        if success:
            logger.info(f"Completed! Saved to {args.output}")
        else:
            logger.error("Processing failed.")
    except Exception as e:
        logger.exception(f"Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
