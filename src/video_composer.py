"""
视频合成模块 - 使用 FFmpeg 进行视频处理和合成
"""

import subprocess
import os
from typing import Optional, List, Dict, Tuple
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class VideoComposer:
    """使用 FFmpeg 进行视频合成"""
    
    def __init__(self, ffmpeg_path: str = 'ffmpeg'):
        """
        初始化视频合成器
        
        Args:
            ffmpeg_path: FFmpeg 可执行文件路径
        """
        self.ffmpeg_path = ffmpeg_path
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """检查 FFmpeg 是否已安装"""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.split('\n')[0]
                logger.info(f"[OK] FFmpeg 已找到: {version}")
            else:
                raise RuntimeError("FFmpeg 返回错误代码")
        except FileNotFoundError:
            logger.error(f"FFmpeg 未找到: {self.ffmpeg_path}")
            logger.info("请安装 FFmpeg:")
            logger.info("  Linux: sudo apt-get install ffmpeg")
            logger.info("  macOS: brew install ffmpeg")
            logger.info("  Windows: 下载 https://ffmpeg.org/download.html")
            raise RuntimeError("FFmpeg not installed")
        except Exception as e:
            logger.error(f"检查 FFmpeg 失败: {e}")
            raise
    
    def get_video_info(self, video_path: str) -> Optional[Dict]:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
        
        Returns:
            视频信息字典，包含 fps、width、height、duration 等
        """
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-print_format', 'json',
                '-show_format',
                '-show_streams'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # FFmpeg 会将信息输出到 stderr
            import json
            # 尝试从输出中提取 JSON（通常在最后）
            output = result.stderr + result.stdout
            
            # 使用 ffprobe 更容易获取信息
            return self._get_video_info_with_ffprobe(video_path)
        
        except Exception as e:
            logger.error(f"获取视频信息失败: {e}")
            return None
    
    def _get_video_info_with_ffprobe(self, video_path: str) -> Optional[Dict]:
        """使用 ffprobe 获取视频信息（更可靠）"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,duration',
                '-of', 'json',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                stream = data['streams'][0]
                
                # 解析帧率
                fps_str = stream.get('r_frame_rate', '30/1')
                num, den = map(int, fps_str.split('/'))
                fps = num / den
                
                return {
                    'width': stream.get('width'),
                    'height': stream.get('height'),
                    'fps': fps,
                    'duration': float(stream.get('duration', 0))
                }
        except Exception as e:
            logger.warning(f"使用 ffprobe 失败: {e}，使用默认值")
        
        # 默认值
        return {
            'width': 1920,
            'height': 1080,
            'fps': 30,
            'duration': 0
        }
    
    def compose_video_with_audio(self, video_path: str, audio_path: str,
                                output_path: str, overwrite: bool = True) -> bool:
        """
        为视频添加音频
        
        Args:
            video_path: 视频文件路径
            audio_path: 音频文件路径
            output_path: 输出视频路径
            overwrite: 是否覆盖输出文件
        
        Returns:
            成功返回 True
        """
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',        # 复制视频流（不重新编码，快速）
                '-c:a', 'aac',         # 音频编码器
                '-b:a', '192k',        # 音频比特率
                '-shortest',           # 以最短流为准
            ]
            
            if overwrite:
                cmd.append('-y')  # 覆盖输出文件
            else:
                cmd.append('-n')  # 不覆盖输出文件
            
            cmd.append(output_path)
            
            logger.info(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"[OK] 视频合成成功: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg 错误: {result.stderr}")
                return False
        
        except subprocess.TimeoutExpired:
            logger.error("视频合成超时")
            return False
        except Exception as e:
            logger.error(f"视频合成失败: {e}")
            return False
    
    def merge_videos(self, video_list: List[str], output_path: str,
                    overwrite: bool = True) -> bool:
        """
        合并多个视频文件
        
        Args:
            video_list: 视频文件路径列表
            output_path: 输出视频路径
            overwrite: 是否覆盖输出文件
        
        Returns:
            成功返回 True
        """
        try:
            # 创建 concat 文件
            concat_file = 'concat_list.txt'
            with open(concat_file, 'w') as f:
                for video in video_list:
                    f.write(f"file '{os.path.abspath(video)}'\n")
            
            cmd = [
                self.ffmpeg_path,
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',           # 直接复制，不重新编码
                '-y' if overwrite else '-n',
                output_path
            ]
            
            logger.info(f"合并 {len(video_list)} 个视频...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # 清理临时文件
            if os.path.exists(concat_file):
                os.remove(concat_file)
            
            if result.returncode == 0:
                logger.info(f"[OK] 视频合并成功: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg 错误: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"视频合并失败: {e}")
            return False
    
    def add_watermark(self, video_path: str, watermark_path: str,
                     output_path: str, position: str = 'top-right',
                     scale: float = 0.1, overwrite: bool = True) -> bool:
        """
        为视频添加水印
        
        Args:
            video_path: 视频文件路径
            watermark_path: 水印图片路径
            output_path: 输出视频路径
            position: 水印位置 ('top-left', 'top-right', 'bottom-left', 'bottom-right', 'center')
            scale: 水印缩放比例（相对于视频宽度）
            overwrite: 是否覆盖输出文件
        
        Returns:
            成功返回 True
        """
        try:
            # 定义位置
            positions = {
                'top-left': 'x=10:y=10',
                'top-right': 'x=main_w-overlay_w-10:y=10',
                'bottom-left': 'x=10:y=main_h-overlay_h-10',
                'bottom-right': 'x=main_w-overlay_w-10:y=main_h-overlay_h-10',
                'center': 'x=(main_w-overlay_w)/2:y=(main_h-overlay_h)/2'
            }
            
            pos = positions.get(position, positions['top-right'])
            
            # 构建 filter_complex
            filter_complex = (
                f"[0:v][1:v]scale2ref=w=iw*{scale}:h=ow/mdar[logo][video];"
                f"[video][logo]overlay={pos}[out]"
            )
            
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-i', watermark_path,
                '-filter_complex', filter_complex,
                '-map', '[out]',
                '-map', '0:a',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-y' if overwrite else '-n',
                output_path
            ]
            
            logger.info(f"添加水印...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"[OK] 水印添加成功: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg 错误: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"水印添加失败: {e}")
            return False
    
    def resize_video(self, video_path: str, output_path: str,
                    width: int = 1920, height: int = 1080,
                    overwrite: bool = True) -> bool:
        """
        调整视频分辨率
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            width: 输出宽度（-1 表示自动）
            height: 输出高度（-1 表示自动）
            overwrite: 是否覆盖输出文件
        
        Returns:
            成功返回 True
        """
        try:
            filter_str = f"scale={width}:{height}"
            
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-vf', filter_str,
                '-c:a', 'aac',
                '-b:a', '192k',
                '-y' if overwrite else '-n',
                output_path
            ]
            
            logger.info(f"调整视频分辨率至 {width}x{height}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info(f"[OK] 分辨率调整成功: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg 错误: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"分辨率调整失败: {e}")
            return False
    
    def convert_format(self, video_path: str, output_path: str,
                      output_format: str = 'mp4', codec: str = 'h264',
                      bitrate: str = '5000k', overwrite: bool = True) -> bool:
        """
        转换视频格式
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            output_format: 输出格式 ('mp4', 'avi', 'mov' 等)
            codec: 视频编码器 ('h264', 'h265', 'vp9' 等)
            bitrate: 比特率 ('5000k', '10M' 等)
            overwrite: 是否覆盖输出文件
        
        Returns:
            成功返回 True
        """
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-c:v', codec,
                '-b:v', bitrate,
                '-c:a', 'aac',
                '-b:a', '192k',
                '-y' if overwrite else '-n',
                output_path
            ]
            
            logger.info(f"转换视频格式为 {output_format} (编码: {codec}, 比特率: {bitrate})...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info(f"[OK] 格式转换成功: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg 错误: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"格式转换失败: {e}")
            return False
    
    def extract_frames(self, video_path: str, output_dir: str,
                      frame_rate: int = 1, start_time: str = '0',
                      duration: str = None) -> bool:
        """
        从视频中提取帧
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            frame_rate: 提取帧率（每秒提取的帧数）
            start_time: 开始时间（HH:MM:SS 或秒数）
            duration: 提取时长（可选）
        
        Returns:
            成功返回 True
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-ss', start_time,
                '-vf', f'fps={frame_rate}',
                '-y',
                os.path.join(output_dir, 'frame_%04d.jpg')
            ]
            
            if duration:
                cmd.insert(4, '-t')
                cmd.insert(5, duration)
            
            logger.info(f"提取视频帧 (每秒 {frame_rate} 帧)...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info(f"[OK] 帧提取成功: {output_dir}")
                return True
            else:
                logger.error(f"FFmpeg 错误: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"帧提取失败: {e}")
            return False


# 快速接口
def quick_compose_video(video_path: str, audio_path: str,
                       output_path: str) -> bool:
    """快速合成视频和音频"""
    composer = VideoComposer()
    return composer.compose_video_with_audio(video_path, audio_path, output_path)


def quick_merge_videos(video_list: List[str], output_path: str) -> bool:
    """快速合并多个视频"""
    composer = VideoComposer()
    return composer.merge_videos(video_list, output_path)


def quick_get_video_info(video_path: str) -> Optional[Dict]:
    """快速获取视频信息"""
    composer = VideoComposer()
    return composer.get_video_info(video_path)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 2:
        if sys.argv[1] == 'compose':
            video = sys.argv[2]
            audio = sys.argv[3]
            output = sys.argv[4] if len(sys.argv) > 4 else 'output.mp4'
            quick_compose_video(video, audio, output)
        
        elif sys.argv[1] == 'info':
            video = sys.argv[2]
            info = quick_get_video_info(video)
            if info:
                print(json.dumps(info, indent=2))
