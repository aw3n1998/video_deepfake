"""
音频处理模块 - TTS（文字转语音）和音频编辑
"""

import numpy as np
import os
from typing import Optional, List, Tuple
import logging
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)


class TTSProcessor:
    """文字转语音处理器"""
    
    def __init__(self, tts_engine: str = 'edge-tts'):
        """
        初始化 TTS 引擎
        
        Args:
            tts_engine: 'edge-tts' | 'pyttsx3' | 'xunfei'
        """
        self.tts_engine = tts_engine
        self.engine = None
        self._init_engine()
    
    def _init_engine(self):
        """初始化对应的 TTS 引擎"""
        if self.tts_engine == 'edge-tts':
            self._init_edge_tts()
        elif self.tts_engine == 'pyttsx3':
            self._init_pyttsx3()
        elif self.tts_engine == 'xunfei':
            self._init_xunfei()
        else:
            raise ValueError(f"Unknown TTS engine: {self.tts_engine}")
    
    def _init_edge_tts(self):
        """初始化 Microsoft Edge TTS（免费，推荐）"""
        try:
            import edge_tts
            self.engine = 'edge-tts'
            logger.info("[OK] Edge TTS 引擎已加载")
        except ImportError:
            logger.error("edge-tts 未安装，请运行: pip install edge-tts")
            raise
    
    def _init_pyttsx3(self):
        """初始化本地 TTS（离线，但质量较差）"""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 200)  # 语速
            logger.info("[OK] pyttsx3 引擎已加载")
        except ImportError:
            logger.error("pyttsx3 未安装，请运行: pip install pyttsx3")
            raise
    
    def _init_xunfei(self):
        """初始化讯飞 TTS（需要 API key）"""
        try:
            import xfyun_speech
            self.engine = xfyun_speech
            logger.info("[OK] 讯飞 TTS 引擎已加载")
        except ImportError:
            logger.error("xfyun-speech 未安装，请运行: pip install xfyun-speech")
            raise
    
    async def text_to_speech_edge(self, text: str, output_path: str,
                                  voice: str = 'zh-CN-XiaoxiaoNeural',
                                  rate: str = '+0%') -> bool:
        """
        使用 Edge TTS 进行文字转语音
        
        Args:
            text: 要转换的文本
            output_path: 输出音频文件路径
            voice: 语音种类
                中文: 'zh-CN-XiaoxiaoNeural' (小晓，女性)
                     'zh-CN-YunjianNeural' (云健，男性)
                英文: 'en-US-AriaNeural' 等
            rate: 语速调整 '+0%' | '+10%' | '-10%' 等
        
        Returns:
            成功返回 True
        """
        try:
            import edge_tts
            
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate
            )
            
            await communicate.save(output_path)
            logger.info(f"[OK] TTS 完成: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Edge TTS 失败: {e}")
            return False
    
    def text_to_speech(self, text: str, output_path: str,
                      voice: str = 'zh-CN-XiaoxiaoNeural') -> bool:
        """
        文字转语音（同步接口）
        """
        if self.tts_engine == 'edge-tts':
            # 运行异步函数
            try:
                asyncio.run(self.text_to_speech_edge(text, output_path, voice))
                return True
            except Exception as e:
                logger.error(f"TTS 失败: {e}")
                return False
        
        elif self.tts_engine == 'pyttsx3':
            try:
                self.engine.save_to_file(text, output_path)
                self.engine.runAndWait()
                logger.info(f"[OK] TTS 完成: {output_path}")
                return True
            except Exception as e:
                logger.error(f"pyttsx3 TTS 失败: {e}")
                return False
        
        return False
    
    def batch_text_to_speech(self, texts: List[str], output_dir: str,
                            voice: str = 'zh-CN-XiaoxiaoNeural') -> List[str]:
        """
        批量文字转语音
        
        Args:
            texts: 文本列表
            output_dir: 输出目录
            voice: 语音种类
        
        Returns:
            生成的音频文件路径列表
        """
        output_paths = []
        os.makedirs(output_dir, exist_ok=True)
        
        for i, text in enumerate(texts):
            output_path = os.path.join(output_dir, f'audio_{i:03d}.mp3')
            success = self.text_to_speech(text, output_path, voice)
            if success:
                output_paths.append(output_path)
        
        logger.info(f"[OK] 生成了 {len(output_paths)} 个音频文件")
        return output_paths


class AudioProcessor:
    """音频处理工具"""
    
    @staticmethod
    def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            (音频数据, 采样率)
        """
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            logger.info(f"[OK] 加载音频: {audio_path} (采样率: {sr} Hz)")
            return y, sr
        except Exception as e:
            logger.error(f"加载音频失败: {e}")
            return None, None
    
    @staticmethod
    def save_audio(audio_data: np.ndarray, output_path: str,
                   sample_rate: int = 44100) -> bool:
        """
        保存音频文件
        
        Args:
            audio_data: 音频数据
            output_path: 输出路径
            sample_rate: 采样率
        
        Returns:
            成功返回 True
        """
        try:
            import soundfile as sf
            sf.write(output_path, audio_data, sample_rate)
            logger.info(f"[OK] 保存音频: {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存音频失败: {e}")
            return False
    
    @staticmethod
    def adjust_volume(audio_data: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """
        调整音量
        
        Args:
            audio_data: 输入音频
            factor: 音量系数 (1.0 = 原始, 2.0 = 双倍, 0.5 = 一半)
        
        Returns:
            处理后的音频
        """
        return audio_data * factor
    
    @staticmethod
    def concatenate_audio(audio_list: List[np.ndarray],
                         sample_rate: int = 44100) -> np.ndarray:
        """
        连接多个音频片段
        
        Args:
            audio_list: 音频片段列表
            sample_rate: 采样率
        
        Returns:
            合并后的音频
        """
        try:
            result = np.concatenate(audio_list)
            logger.info(f"[OK] 合并了 {len(audio_list)} 个音频片段")
            return result
        except Exception as e:
            logger.error(f"合并音频失败: {e}")
            return None
    
    @staticmethod
    def extract_audio_from_video(video_path: str, output_audio_path: str) -> bool:
        """
        从视频中提取音频
        
        Args:
            video_path: 视频文件路径
            output_audio_path: 输出音频路径
        
        Returns:
            成功返回 True
        """
        try:
            import subprocess
            
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # 不处理视频
                '-acodec', 'libmp3lame',
                '-b:a', '192k',
                output_audio_path,
                '-y'  # 覆盖输出文件
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"[OK] 音频提取成功: {output_audio_path}")
                return True
            else:
                logger.error(f"FFmpeg 错误: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"音频提取失败: {e}")
            return False
    
    @staticmethod
    def add_audio_to_video(video_path: str, audio_path: str,
                          output_path: str, audio_volume: float = 1.0) -> bool:
        """
        为视频添加音频
        
        Args:
            video_path: 视频文件路径
            audio_path: 音频文件路径
            output_path: 输出视频路径
            audio_volume: 音量倍数
        
        Returns:
            成功返回 True
        """
        try:
            import subprocess
            
            # FFmpeg 命令来添加音频
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',  # 复制视频流（不重新编码）
                '-c:a', 'aac',   # 重新编码音频
                '-b:a', '192k',
                '-filter:a', f'volume={audio_volume}',  # 调整音量
                '-shortest',  # 以最短的流为准
                output_path,
                '-y'  # 覆盖输出文件
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"[OK] 音频合成成功: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg 错误: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"音频合成失败: {e}")
            return False
    
    @staticmethod
    def mix_audio(primary_audio_path: str, secondary_audio_path: str,
                  output_path: str, primary_volume: float = 1.0,
                  secondary_volume: float = 0.5) -> bool:
        """
        混音两个音频
        
        Args:
            primary_audio_path: 主音频路径
            secondary_audio_path: 副音频路径
            output_path: 输出音频路径
            primary_volume: 主音频音量
            secondary_volume: 副音频音量
        
        Returns:
            成功返回 True
        """
        try:
            import subprocess
            
            filter_complex = (
                f'[0:a]volume={primary_volume}[a1];'
                f'[1:a]volume={secondary_volume}[a2];'
                f'[a1][a2]amix=inputs=2:duration=longest[a]'
            )
            
            cmd = [
                'ffmpeg',
                '-i', primary_audio_path,
                '-i', secondary_audio_path,
                '-filter_complex', filter_complex,
                '-map', '[a]',
                '-c:a', 'aac',
                '-b:a', '192k',
                output_path,
                '-y'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"[OK] 混音成功: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg 错误: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"混音失败: {e}")
            return False


class ASRProcessor:
    """
    语音识别处理器（使用 faster-whisper）
    安装: pip install faster-whisper
    首次运行会自动下载对应尺寸的模型（small ~500MB）
    """

    def __init__(self, model_size: str = 'small', device: str = 'cpu'):
        """
        Args:
            model_size: 'tiny' | 'base' | 'small' | 'medium' | 'large-v2'
                        small 在中文上精度和速度均衡，推荐使用
            device: 'cpu' 或 'cuda'（有 GPU 时自动加速）
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from faster_whisper import WhisperModel
            compute_type = 'int8' if self.device == 'cpu' else 'float16'
            self.model = WhisperModel(
                self.model_size, device=self.device, compute_type=compute_type
            )
            logger.info(f"[OK] faster-whisper 加载成功 (model={self.model_size}, device={self.device})")
        except ImportError:
            logger.error("faster-whisper 未安装，请运行: pip install faster-whisper")
            raise

    def transcribe(self, audio_path: str, language: str = 'zh') -> str:
        """
        转录音频文件为文字

        Args:
            audio_path: 音频文件路径（支持 wav/mp3/m4a 等）
            language: 语言代码，'zh'=中文, 'en'=英文, None=自动检测

        Returns:
            转录文本字符串
        """
        try:
            segments, info = self.model.transcribe(
                audio_path, language=language, beam_size=5
            )
            transcript = ''.join(seg.text for seg in segments)
            logger.info(
                f"[OK] 语音识别完成 | 语言: {info.language} "
                f"| 文本长度: {len(transcript)} 字"
            )
            return transcript
        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            return ''

    def transcribe_video(self, video_path: str, language: str = 'zh') -> str:
        """
        直接从视频提取音频并转录

        Args:
            video_path: 视频文件路径
            language: 语言代码

        Returns:
            转录文本字符串
        """
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_audio = tmp.name

        try:
            # 提取为 16kHz 单声道 wav，Whisper 最佳输入格式
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-ar', '16000', '-ac', '1',
                '-y', tmp_audio
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode != 0:
                logger.error("从视频提取音频失败")
                return ''
            logger.info(f"[OK] 音频提取完成，开始识别...")
            return self.transcribe(tmp_audio, language)
        finally:
            if os.path.exists(tmp_audio):
                os.remove(tmp_audio)


# 快速接口
def quick_tts(text: str, output_path: str, voice: str = 'zh-CN-XiaoxiaoNeural') -> bool:
    """快速文字转语音"""
    tts = TTSProcessor('edge-tts')
    return tts.text_to_speech(text, output_path, voice)


def quick_extract_audio(video_path: str, output_audio_path: str) -> bool:
    """快速提取视频音频"""
    return AudioProcessor.extract_audio_from_video(video_path, output_audio_path)


def quick_add_audio_to_video(video_path: str, audio_path: str,
                             output_path: str) -> bool:
    """快速为视频添加音频"""
    return AudioProcessor.add_audio_to_video(video_path, audio_path, output_path)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 2:
        if sys.argv[1] == 'tts':
            text = sys.argv[2]
            output = sys.argv[3] if len(sys.argv) > 3 else 'output.mp3'
            quick_tts(text, output)
        
        elif sys.argv[1] == 'extract':
            video = sys.argv[2]
            audio = sys.argv[3] if len(sys.argv) > 3 else 'audio.mp3'
            quick_extract_audio(video, audio)
