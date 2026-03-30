"""
AI 视频智能重绘工具

模块:
  - vid2vid: 全局视频风格重绘 (img2img + ControlNet)
  - person_swap: 指定人物替换 (Inpainting + IP-Adapter + 人脸追踪)
  - video_io: 视频 I/O 工具
  - utils: 通用安全工具
"""

from .vid2vid import Vid2VidPipeline
from .person_swap import PersonSwapPipeline
from .video_io import get_video_info, merge_video_audio, check_ffmpeg
from .utils import ensure_dir, get_timestamp, validate_video_path, validate_image_path

__version__ = "2.1.0"
__all__ = [
    "Vid2VidPipeline",
    "PersonSwapPipeline",
    "get_video_info",
    "merge_video_audio",
    "check_ffmpeg",
    "ensure_dir",
    "get_timestamp",
    "validate_video_path",
    "validate_image_path",
]
