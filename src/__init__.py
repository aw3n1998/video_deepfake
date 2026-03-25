"""
Video Deepfake Processor - 完整的视频深度伪造处理框架

模块结构:
  - face_detector: 人脸检测
  - face_swapper: 人脸交换（Deepfaceslive / InsightFace）
  - audio_processor: TTS 和音频处理
  - video_composer: FFmpeg 视频合成
  - pipeline: 完整工作流程编排
"""

from .face_detector import FaceDetector, quick_detect, quick_detect_video
from .face_swapper import FaceSwapper, DeepfaceLiveSwapper, quick_face_swap, quick_face_swap_video
from .audio_processor import TTSProcessor, AudioProcessor, quick_tts, quick_extract_audio, quick_add_audio_to_video
from .video_composer import VideoComposer, quick_compose_video, quick_merge_videos, quick_get_video_info
from .pipeline import VideoDeepfakePipeline

__version__ = '1.0.0'
__all__ = [
    # 人脸检测
    'FaceDetector',
    'quick_detect',
    'quick_detect_video',
    
    # 人脸交换
    'FaceSwapper',
    'DeepfaceLiveSwapper',
    'quick_face_swap',
    'quick_face_swap_video',
    
    # 音频处理
    'TTSProcessor',
    'AudioProcessor',
    'quick_tts',
    'quick_extract_audio',
    'quick_add_audio_to_video',
    
    # 视频合成
    'VideoComposer',
    'quick_compose_video',
    'quick_merge_videos',
    'quick_get_video_info',
    
    # 完整工作流
    'VideoDeepfakePipeline',
]
