"""
视频 I/O 工具模块

安全的 FFmpeg 封装，替代原 video_composer.py 中的冗余功能。
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def check_ffmpeg() -> bool:
    """检查 FFmpeg 是否可用"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_video_info(video_path: str) -> Optional[Dict]:
    """
    使用 ffprobe 获取视频信息。

    Returns:
        {'width': int, 'height': int, 'fps': float, 'duration': float, 'has_audio': bool}
    """
    if not os.path.isfile(video_path):
        return None

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,duration",
                "-of", "json",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        stream = data["streams"][0]
        fps_str = stream.get("r_frame_rate", "30/1")
        num, den = map(int, fps_str.split("/"))
        fps = num / den if den > 0 else 30.0

        # 检查是否有音频流
        audio_check = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "json",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        has_audio = '"codec_type"' in audio_check.stdout

        return {
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "fps": fps,
            "duration": float(stream.get("duration", 0)),
            "has_audio": has_audio,
        }
    except Exception as e:
        logger.warning(f"获取视频信息失败: {e}")
        return None


def merge_video_audio(
    video_path: str,
    audio_source: str,
    output_path: str,
) -> bool:
    """
    将视频与音频源合并。

    Args:
        video_path: 视频文件路径
        audio_source: 音频来源（视频或音频文件）
        output_path: 输出路径
    """
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", video_path,
                "-i", audio_source,
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                "-y", output_path,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            logger.info(f"音频合并成功: {output_path}")
            return True
        else:
            logger.error(f"FFmpeg 错误: {result.stderr[:500]}")
            return False
    except Exception as e:
        logger.error(f"音频合并失败: {e}")
        return False
