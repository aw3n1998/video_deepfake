"""
完整视频深度伪造工作流程编排
流程: 视频输入 -> 换脸 -> TTS -> 音频处理 -> 视频合成 -> 输出
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, List
import logging
from datetime import datetime

from .face_detector import FaceDetector
from .face_swapper import FaceSwapper
from .audio_processor import TTSProcessor, AudioProcessor
from .video_composer import VideoComposer
from .utils import ensure_dir, get_timestamp

# 配置日志（确保 logs 目录存在）
Path('logs').mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VideoDeepfakePipeline:
    """完整的视频深度伪造工作流"""
    
    def __init__(self, config: Dict = None):
        """
        初始化工作流
        
        Args:
            config: 配置字典，包含各模块的参数
        """
        self.config = config or self._load_default_config()
        self.work_dir = Path(self.config.get('work_dir', './data'))
        self._init_components()
        self._create_directories()
    
    def _load_default_config(self) -> Dict:
        """加载默认配置"""
        return {
            'work_dir': './data',
            'face_detector': {
                'model_type': 'mtcnn',
                'confidence_threshold': 0.9
            },
            'face_swapper': {
                'model_name': 'inswapper_128.onnx'
            },
            'tts': {
                'engine': 'edge-tts',
                'voice': 'zh-CN-XiaoxiaoNeural',
                'rate': '+0%'
            },
            'video': {
                'output_format': 'mp4',
                'bitrate': '5000k',
                'codec': 'h264'
            }
        }
    
    def _init_components(self):
        """初始化各个处理模块"""
        logger.info("初始化处理模块...")
        
        try:
            self.face_detector = FaceDetector(
                model_type=self.config['face_detector']['model_type']
            )
            self.face_swapper = FaceSwapper(
                model_name=self.config['face_swapper']['model_name']
            )
            self.tts_processor = TTSProcessor(
                tts_engine=self.config['tts']['engine']
            )
            self.audio_processor = AudioProcessor()
            self.video_composer = VideoComposer()
            
            logger.info("[OK] 所有模块初始化成功")
        except Exception as e:
            logger.error(f"模块初始化失败: {e}")
            raise
    
    def _create_directories(self):
        """创建工作目录"""
        dirs = [
            self.work_dir / 'input',
            self.work_dir / 'output',
            self.work_dir / 'cache',
            self.work_dir / 'logs'
        ]
        
        for dir_path in dirs:
            ensure_dir(dir_path)
    
    def process_full_pipeline(self, 
                             source_image: str,
                             input_video: str,
                             output_text: str,
                             output_path: str) -> bool:
        """
        完整工作流程
        
        Args:
            source_image: 源人脸图片路径
            input_video: 输入视频路径
            output_text: 输出文字（用于 TTS）
            output_path: 输出视频路径
        
        Returns:
            成功返回 True
        """
        logger.info("=" * 60)
        logger.info("开始视频深度伪造处理")
        logger.info("=" * 60)
        
        try:
            # 第一步: 人脸交换
            logger.info("\n[第一步] 进行人脸交换...")
            swapped_video = self._step_face_swap(source_image, input_video)
            if not swapped_video:
                return False
            
            # 第二步: 生成语音
            logger.info("\n[第二步] 生成语音...")
            audio_file = self._step_tts(output_text)
            if not audio_file:
                return False
            
            # 第三步: 合成视频和音频
            logger.info("\n[第三步] 合成视频和音频...")
            success = self._step_compose(swapped_video, audio_file, output_path)
            
            if success:
                logger.info("\n" + "=" * 60)
                logger.info("[OK] 处理完成！")
                logger.info(f"输出文件: {output_path}")
                logger.info("=" * 60)
                return True
            else:
                return False
        
        except Exception as e:
            logger.error(f"工作流执行失败: {e}")
            return False
    
    def _step_face_swap(self, source_image: str, input_video: str) -> Optional[str]:
        """
        第一步: 人脸交换
        """
        try:
            cache_dir = self.work_dir / 'cache'
            output_video = cache_dir / 'swapped_video.mp4'
            
            logger.info(f"源图片: {source_image}")
            logger.info(f"输入视频: {input_video}")
            logger.info(f"输出视频: {output_video}")
            
            success = self.face_swapper.swap_faces_in_video(
                str(source_image),
                str(input_video),
                str(output_video)
            )
            
            if success:
                logger.info(f"[OK] 人脸交换完成: {output_video}")
                return str(output_video)
            else:
                logger.error("人脸交换失败")
                return None
        
        except Exception as e:
            logger.error(f"人脸交换错误: {e}")
            return None
    
    def _step_tts(self, text: str) -> Optional[str]:
        """
        第二步: 文字转语音
        """
        try:
            cache_dir = self.work_dir / 'cache'
            audio_file = cache_dir / 'generated_audio.mp3'
            
            logger.info(f"生成文本: {text[:50]}...")
            
            success = self.tts_processor.text_to_speech(
                text=text,
                output_path=str(audio_file),
                voice=self.config['tts']['voice']
            )
            
            if success:
                logger.info(f"[OK] 语音生成完成: {audio_file}")
                return str(audio_file)
            else:
                logger.error("语音生成失败")
                return None
        
        except Exception as e:
            logger.error(f"TTS 错误: {e}")
            return None
    
    def _step_compose(self, video_path: str, audio_path: str,
                     output_path: str) -> bool:
        """
        第三步: 视频和音频合成
        """
        try:
            logger.info(f"视频: {video_path}")
            logger.info(f"音频: {audio_path}")
            logger.info(f"输出: {output_path}")
            
            success = self.video_composer.compose_video_with_audio(
                video_path=video_path,
                audio_path=audio_path,
                output_path=output_path,
                overwrite=True
            )
            
            if success:
                logger.info(f"[OK] 视频合成完成: {output_path}")
                return True
            else:
                logger.error("视频合成失败")
                return False
        
        except Exception as e:
            logger.error(f"视频合成错误: {e}")
            return False
    
    def process_with_subtitles(self, 
                              source_image: str,
                              input_video: str,
                              output_text: str,
                              output_path: str,
                              add_subtitles: bool = True) -> bool:
        """
        完整工作流程（包含字幕）
        
        Args:
            source_image: 源人脸图片
            input_video: 输入视频
            output_text: 输出文字
            output_path: 输出视频
            add_subtitles: 是否添加字幕
        
        Returns:
            成功返回 True
        """
        # 先进行基本处理
        success = self.process_full_pipeline(
            source_image, input_video, output_text, output_path
        )
        
        if success and add_subtitles:
            # 添加字幕
            logger.info("\n[第四步] 添加字幕...")
            output_with_subs = output_path.replace('.mp4', '_with_subtitles.mp4')
            self._add_subtitles(output_path, output_text, output_with_subs)
            
            return True
        
        return success
    
    def _add_subtitles(self, video_path: str, text: str, output_path: str) -> bool:
        """
        为视频添加字幕
        """
        try:
            # 创建 SRT 字幕文件
            srt_path = str(Path(output_path).parent / 'subtitles.srt')
            
            # 简单的字幕格式：整个视频时间内显示文本
            srt_content = f"""1
00:00:00,000 --> 00:59:59,999
{text}
"""
            
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            logger.info(f"[OK] 字幕文件已生成: {srt_path}")
            
            # 使用 FFmpeg 添加字幕
            import subprocess
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', f"subtitles='{srt_path}'",
                '-c:a', 'copy',
                '-y',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info(f"[OK] 字幕已添加: {output_path}")
                return True
            else:
                logger.warning(f"字幕添加失败: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"添加字幕错误: {e}")
            return False
    
    def batch_process(self, config_list: List[Dict]) -> Dict:
        """
        批量处理多个视频
        
        Args:
            config_list: 处理配置列表，每个配置包含:
                {
                    'source_image': '...',
                    'input_video': '...',
                    'output_text': '...',
                    'output_path': '...'
                }
        
        Returns:
            处理结果字典
        """
        results = {
            'success': 0,
            'failed': 0,
            'details': []
        }
        
        for i, config in enumerate(config_list):
            logger.info(f"\n处理任务 {i+1}/{len(config_list)}")
            
            success = self.process_full_pipeline(
                source_image=config['source_image'],
                input_video=config['input_video'],
                output_text=config['output_text'],
                output_path=config['output_path']
            )
            
            result_detail = {
                'task_id': i + 1,
                'success': success,
                'config': config,
                'timestamp': datetime.now().isoformat()
            }
            
            results['details'].append(result_detail)
            
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
        
        # 保存处理结果
        result_file = self.work_dir / 'logs' / f"batch_result_{get_timestamp()}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n批处理完成: {results['success']} 成功, {results['failed']} 失败")
        logger.info(f"结果已保存: {result_file}")
        
        return results


def main():
    """主函数示例"""
    import sys
    
    if len(sys.argv) < 4:
        print("使用方法:")
        print("  python -m src.pipeline <source_image> <input_video> <output_text> [output_path]")
        print("\n示例:")
        print("  python -m src.pipeline source.jpg video.mp4 '你好世界' output.mp4")
        sys.exit(1)
    
    source_image = sys.argv[1]
    input_video = sys.argv[2]
    output_text = sys.argv[3]
    output_path = sys.argv[4] if len(sys.argv) > 4 else 'output.mp4'
    
    # 初始化工作流
    pipeline = VideoDeepfakePipeline()
    
    # 执行完整工作流
    success = pipeline.process_full_pipeline(
        source_image=source_image,
        input_video=input_video,
        output_text=output_text,
        output_path=output_path
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
