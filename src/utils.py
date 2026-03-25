"""
工具函数库
"""

import os
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def ensure_dir(directory: Path) -> Path:
    """确保目录存在"""
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_timestamp() -> str:
    """获取当前时间戳"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def get_file_size(file_path: str) -> str:
    """获取文件大小（可读格式）"""
    size = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}TB"


def clean_cache_dir(cache_dir: Path, keep_latest: int = 3):
    """
    清理缓存目录，只保留最新的 N 个文件
    """
    try:
        files = sorted(
            cache_dir.glob('*'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for file in files[keep_latest:]:
            if file.is_file():
                file.unlink()
                logger.info(f"删除缓存文件: {file.name}")
    
    except Exception as e:
        logger.warning(f"清理缓存失败: {e}")


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}


def save_config(config: dict, output_path: str):
    """保存配置到 YAML 文件"""
    try:
        import yaml
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        logger.info(f"配置已保存: {output_path}")
    except Exception as e:
        logger.error(f"保存配置失败: {e}")
