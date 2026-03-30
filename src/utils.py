"""
工具函数库 - 包含安全工具
"""

import os
import re
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def ensure_dir(directory) -> Path:
    """确保目录存在"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_timestamp() -> str:
    """获取当前时间戳"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_file_size(file_path: str) -> str:
    """获取文件大小（可读格式）"""
    size = os.path.getsize(file_path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}TB"


def validate_video_path(path: str) -> bool:
    """验证视频文件路径安全性"""
    if not path:
        return False

    p = Path(path).resolve()

    # 检查文件存在性
    if not p.is_file():
        logger.warning(f"文件不存在: {path}")
        return False

    # 检查扩展名白名单
    allowed_ext = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
    if p.suffix.lower() not in allowed_ext:
        logger.warning(f"不支持的视频格式: {p.suffix}")
        return False

    return True


def validate_image_path(path: str) -> bool:
    """验证图片文件路径安全性"""
    if not path:
        return False

    p = Path(path).resolve()

    if not p.is_file():
        logger.warning(f"文件不存在: {path}")
        return False

    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    if p.suffix.lower() not in allowed_ext:
        logger.warning(f"不支持的图片格式: {p.suffix}")
        return False

    return True


def sanitize_prompt(prompt: str, max_length: int = 1000) -> str:
    """清理提示词，移除潜在危险字符"""
    if not prompt:
        return ""
    # 移除控制字符，保留基本文本
    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', prompt)
    return cleaned[:max_length].strip()


def clamp(value: float, min_val: float, max_val: float) -> float:
    """将值限制在范围内"""
    return max(min_val, min(max_val, value))
