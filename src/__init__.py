"""
AI 视频重绘工具

模块结构:
  - vid2vid: 核心视频转视频管线 (img2img + ControlNet)
  - video_io: 视频 I/O 和 FFmpeg 工具
  - utils: 通用工具函数
"""

from .vid2vid import Vid2VidPipeline
from .video_io import get_video_info, merge_video_audio, check_ffmpeg
from .utils import ensure_dir, get_timestamp, validate_video_path, validate_image_path

__version__ = "2.0.0"
__all__ = [
    "Vid2VidPipeline",
    "get_video_info",
    "merge_video_audio",
    "check_ffmpeg",
    "ensure_dir",
    "get_timestamp",
    "validate_video_path",
    "validate_image_path",
]
