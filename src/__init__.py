"""
AI 视频智能重绘工具

模块:
  - vid2vid: 全局视频风格重绘 (img2img + ControlNet)
  - vid2vid_gen: Wan2.1 视频内容生成 (动作/物体/场景级改变)
  - person_swap: 指定人物替换 (Inpainting + IP-Adapter + 人脸追踪)
  - prompt_router: 提示词意图路由 (风格 vs 内容)
  - video_io: 视频 I/O 工具
  - utils: 通用安全工具
"""

from .vid2vid import Vid2VidPipeline
from .vid2vid_gen import Wan2VidPipeline
from .person_swap import PersonSwapPipeline
from .prompt_router import classify_prompt, route_pipeline
from .hair_effect import HairFallEffect, should_trigger as hair_trigger, is_hair_only
from .video_io import get_video_info, merge_video_audio, check_ffmpeg
from .utils import ensure_dir, get_timestamp, validate_video_path, validate_image_path

__version__ = "3.0.0"
__all__ = [
    "Vid2VidPipeline",
    "Wan2VidPipeline",
    "PersonSwapPipeline",
    "classify_prompt",
    "route_pipeline",
    "get_video_info",
    "merge_video_audio",
    "check_ffmpeg",
    "ensure_dir",
    "get_timestamp",
    "validate_video_path",
    "validate_image_path",
]
