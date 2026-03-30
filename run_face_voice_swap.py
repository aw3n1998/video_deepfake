#!/usr/bin/env python3
"""
一键处理：检测并替换视频中的人像内容，支持声音变调。
完全通用化，不再硬编码性别转换逻辑。

处理流程:
  1. 逐帧检测人脸（支持性别过滤或全部替换），替换为指定目标人脸。
  2. 用 faster-whisper 识别原视频中的语音内容。
  3. 用 edge-tts 生成对应音色的音频。
  4. FFmpeg 合并替换后的视频画面 + 新音频。

用法:
  python run_face_voice_swap.py --input 视频.mp4 --face 目标脸.jpg --output 输出.mp4
"""

import argparse
import os
import shutil
import sys
import logging
from pathlib import Path

# 把项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent))

from src.face_swapper import FaceSwapper
from src.audio_processor import ASRProcessor, TTSProcessor
from src.video_composer import VideoComposer
from src.utils import ensure_dir, get_timestamp

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def process(input_video: str, target_face: str, output_path: str,
            tts_voice: str = 'zh-CN-XiaoxiaoNeural',
            asr_lang: str = 'zh',
            filter_gender: int = -1,
            keep_temp: bool = False,
            swapped_video: str = None,
            workers: int = 1) -> bool:

    if not os.environ.get('HF_ENDPOINT'):
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    work_dir = Path('data') / f'temp_vswap_{get_timestamp()}'
    ensure_dir(work_dir)

    logger.info("=" * 60)
    logger.info("  通用人工智能人像与声音转换  ")
    logger.info("=" * 60)
    logger.info(f"输入视频 : {input_video}")
    logger.info(f"目标人脸 : {target_face}")
    logger.info(f"输出路径 : {output_path}")
    logger.info(f"性别过滤 : {'男' if filter_gender==1 else '女' if filter_gender==0 else '全换'}")
    logger.info("=" * 60)

    try:
        # ----------------------------------------------------------------
        # 步骤 1：替换人脸
        # ----------------------------------------------------------------
        if swapped_video:
            swapped_no_audio = swapped_video
            logger.info(f"[步骤 1/3] 跳过（使用已有文件: {swapped_no_audio})")
        else:
            logger.info("\n[步骤 1/3] 检测并替换人脸...")
            swapped_no_audio = str(work_dir / 'swapped_noaudio.mp4')

            swapper = FaceSwapper()
            if workers > 1:
                success = swapper.swap_faces_in_video_parallel(
                    source_image_path=target_face,
                    video_path=input_video,
                    output_path=swapped_no_audio,
                    filter_gender=filter_gender,
                    n_workers=workers
                )
            else:
                success = swapper.swap_faces_by_gender(
                    source_image_path=target_face,
                    video_path=input_video,
                    output_path=swapped_no_audio,
                    filter_gender=filter_gender
                )
            if not success:
                logger.error("[FAIL] 人脸替换失败")
                return False

        # ----------------------------------------------------------------
        # 步骤 2：语音处理
        # ----------------------------------------------------------------
        logger.info("\n[步骤 2/3] 语音识别 + 新声音生成...")
        asr = ASRProcessor(model_size='small', device='cpu')
        transcript = asr.transcribe_video(input_video, language=asr_lang)

        if not transcript.strip():
            logger.warning("未识别到语音，生成静音轨道")
            new_audio = str(work_dir / 'silence.mp3')
            import subprocess
            subprocess.run(['ffmpeg', '-f', 'lavfi', '-i', 'anullsrc', '-t', '1', '-y', new_audio], capture_output=True)
        else:
            new_audio = str(work_dir / 'new_voice.mp3')
            tts = TTSProcessor('edge-tts')
            ok = tts.text_to_speech(transcript, new_audio, voice=tts_voice)
            if not ok: return False

        # ----------------------------------------------------------------
        # 步骤 3：合并
        # ----------------------------------------------------------------
        logger.info("\n[步骤 3/3] 合并最终视频...")
        ensure_dir(Path(output_path).parent)
        composer = VideoComposer()
        success = composer.compose_video_with_audio(swapped_no_audio, new_audio, output_path, overwrite=True)
        
        return success

    except Exception as e:
        logger.exception(f"处理失败: {e}")
        return False
    finally:
        if not keep_temp and work_dir.exists() and Path(output_path).exists():
            shutil.rmtree(work_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description='通用：人脸替换 + 声音转换')
    parser.add_argument('--input',  required=True, help='输入视频路径')
    parser.add_argument('--face',   required=True, help='目标人脸图片路径')
    parser.add_argument('--output', default='output_swapped.mp4', help='输出视频路径')
    parser.add_argument('--gender', type=int, default=-1, help='过滤性别 (1:男, 0:女, -1:全换)')
    parser.add_argument('--voice',  default='zh-CN-XiaoxiaoNeural', help='TTS 音色')
    parser.add_argument('--lang',   default='zh', help='ASR 语言')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--keep-temp', action='store_true')

    args = parser.parse_args()

    success = process(
        input_video=args.input, target_face=args.face, output_path=args.output,
        tts_voice=args.voice, asr_lang=args.lang, filter_gender=args.gender,
        keep_temp=args.keep_temp, workers=args.workers
    )
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
