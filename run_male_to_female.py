#!/usr/bin/env python3
"""
一键处理：检测视频中所有男性人脸替换成女性人脸，并将声音转换成女声。

处理流程:
  1. 逐帧检测男性人脸（InsightFace gender 属性），替换为指定女性人脸
  2. 用 faster-whisper 识别原视频中的语音内容
  3. 用 edge-tts 生成对应的女声音频
  4. FFmpeg 合并替换后的视频画面 + 女声音频

用法:
  python run_male_to_female.py --input 视频.mp4 --face 女性人脸.jpg --output 输出.mp4

参数:
  --input   输入视频路径（必填）
  --face    替换用女性人脸图片路径（必填）
  --output  输出视频路径（默认: output_female.mp4）
  --voice   TTS 音色（默认: zh-CN-XiaoxiaoNeural，可选见文末）
  --lang    ASR 语言（默认: zh，英文用 en）
  --keep-temp  保留临时文件（调试用）

可用女声音色（部分）:
  zh-CN-XiaoxiaoNeural   小晓（温柔，推荐）
  zh-CN-XiaoyiNeural     小艺（活泼）
  zh-CN-XiaohanNeural    小涵（成熟）
  zh-TW-HsiaoChenNeural  台湾女声
  en-US-AriaNeural       英文女声
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


def process(input_video: str, female_face: str, output_path: str,
            tts_voice: str = 'zh-CN-XiaoxiaoNeural',
            asr_lang: str = 'zh',
            keep_temp: bool = False,
            swapped_video: str = None,
            workers: int = 1) -> bool:

    # 使用国内镜像下载 HuggingFace 模型，避免代理问题
    if not os.environ.get('HF_ENDPOINT'):
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    work_dir = Path('data') / f'temp_{get_timestamp()}'
    ensure_dir(work_dir)

    logger.info("=" * 60)
    logger.info("  男性人脸替换 + 声音女声化  ")
    logger.info("=" * 60)
    logger.info(f"输入视频 : {input_video}")
    logger.info(f"女性人脸 : {female_face}")
    logger.info(f"输出路径 : {output_path}")
    logger.info(f"TTS 音色 : {tts_voice}")
    logger.info(f"ASR 语言 : {asr_lang}")
    logger.info("=" * 60)

    try:
        # ----------------------------------------------------------------
        # 步骤 1：替换男性人脸（输出无音频的视频）
        # ----------------------------------------------------------------
        if swapped_video:
            # 直接使用已有的换脸视频，跳过步骤1
            swapped_no_audio = swapped_video
            logger.info(f"[步骤 1/3] 跳过（使用已有文件: {swapped_no_audio})")
        else:
            logger.info("\n[步骤 1/3] 检测并替换男性人脸...")
            swapped_no_audio = str(work_dir / 'swapped_noaudio.mp4')

            swapper = FaceSwapper()
            if workers > 1:
                logger.info(f"使用并行模式（{workers} 进程）")
                success = swapper.swap_male_faces_in_video_parallel(
                    source_image_path=female_face,
                    video_path=input_video,
                    output_path=swapped_no_audio,
                    n_workers=workers
                )
            else:
                success = swapper.swap_male_faces_in_video(
                    source_image_path=female_face,
                    video_path=input_video,
                    output_path=swapped_no_audio
                )
            if not success:
                logger.error("[FAIL] 人脸替换失败，流程中断")
                return False
            logger.info(f"[OK] 步骤 1 完成 -> {swapped_no_audio}")

        # ----------------------------------------------------------------
        # 步骤 2：语音识别 → 生成女声音频
        # ----------------------------------------------------------------
        logger.info("\n[步骤 2/3] 语音识别 + 女声生成...")

        # ASR：识别原视频台词
        asr = ASRProcessor(model_size='small', device='cpu')
        transcript = asr.transcribe_video(input_video, language=asr_lang)

        if not transcript.strip():
            logger.warning("未识别到语音内容，将使用静音音频")
            # 生成静音占位音频，保证后续 FFmpeg 合成不报错
            female_audio = str(work_dir / 'silence.mp3')
            import subprocess
            subprocess.run(
                ['ffmpeg', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
                 '-t', '1', '-y', female_audio],
                capture_output=True
            )
        else:
            logger.info(f"识别文本（前100字）: {transcript[:100]}...")
            female_audio = str(work_dir / 'female_voice.mp3')
            tts = TTSProcessor('edge-tts')
            ok = tts.text_to_speech(transcript, female_audio, voice=tts_voice)
            if not ok:
                logger.error("[FAIL] TTS 生成女声失败，流程中断")
                return False

        logger.info(f"[OK] 步骤 2 完成 -> {female_audio}")

        # ----------------------------------------------------------------
        # 步骤 3：合并替换后的视频画面 + 女声音频
        # ----------------------------------------------------------------
        logger.info("\n[步骤 3/3] 合并视频画面与女声音频...")

        # 确保输出目录存在
        out_dir = Path(output_path).parent
        if str(out_dir) != '.':
            ensure_dir(out_dir)

        composer = VideoComposer()
        success = composer.compose_video_with_audio(
            video_path=swapped_no_audio,
            audio_path=female_audio,
            output_path=output_path,
            overwrite=True
        )
        if not success:
            logger.error("[FAIL] 视频合成失败")
            return False

        logger.info(f"[OK] 步骤 3 完成 -> {output_path}")
        logger.info("\n" + "=" * 60)
        logger.info(f"  处理完成！输出文件: {output_path}  ")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.exception(f"处理过程中发生错误: {e}")
        logger.info(f"临时文件保留在: {work_dir}（可用 --swapped-video 断点续跑）")
        return False

    finally:
        # 只有成功且未指定 keep-temp 时才清理
        if not keep_temp and work_dir.exists():
            # 检查输出文件是否真的生成了，生成了才清理
            if Path(output_path).exists():
                shutil.rmtree(work_dir, ignore_errors=True)
                logger.info(f"临时文件已清理: {work_dir}")
            else:
                logger.info(f"输出文件未生成，临时文件保留在: {work_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='一键：男性人脸替换 + 声音改女声',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input',  required=True, help='输入视频路径')
    parser.add_argument('--face',   required=True, help='替换用女性人脸图片路径')
    parser.add_argument('--output', default='output_female.mp4', help='输出视频路径')
    parser.add_argument('--voice',  default='zh-CN-XiaoxiaoNeural', help='TTS 女声音色')
    parser.add_argument('--lang',   default='zh', help='ASR 语言代码（zh/en）')
    parser.add_argument('--keep-temp', action='store_true', help='保留临时文件（调试用）')
    parser.add_argument('--swapped-video', default=None, help='跳过步骤1，直接使用已换脸的视频文件（断点续跑用）')
    parser.add_argument('--workers', type=int, default=1, help='并行处理进程数（默认1=单进程，建议设为CPU核心数的一半，如4）')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[FAIL] 输入视频不存在: {args.input}")
        sys.exit(1)
    if not os.path.exists(args.face):
        print(f"[FAIL] 女性人脸图片不存在: {args.face}")
        sys.exit(1)
    if args.swapped_video and not os.path.exists(args.swapped_video):
        print(f"[FAIL] 换脸视频不存在: {args.swapped_video}")
        sys.exit(1)

    success = process(
        input_video=args.input,
        female_face=args.face,
        output_path=args.output,
        tts_voice=args.voice,
        asr_lang=args.lang,
        keep_temp=args.keep_temp,
        swapped_video=args.swapped_video,
        workers=args.workers
    )
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
