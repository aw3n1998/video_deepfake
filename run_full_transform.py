#!/usr/bin/env python3
"""
完整转换：男性 -> 女性（全身重绘 + 换脸 + 换声音）
专为 GPU 环境设计（Google Colab T4 / 本地显卡）

处理流程:
  1. ControlNet + SD  全身重绘（男性外观 -> 女性外观）
  2. InsightFace      换脸精修（在重绘结果上叠加真实女脸）
  3. faster-whisper   识别原视频语音
  4. edge-tts         合成女声音频
  5. FFmpeg           合并视频画面 + 女声输出成品

用法:
  python run_full_transform.py --input 视频.mp4 --face 女性人脸.png --output 输出.mp4

参数:
  --input        输入视频路径（必填）
  --face         替换用女性人脸图片（必填，InsightFace 换脸用）
  --output       输出视频路径（默认 output_full.mp4）
  --voice        TTS 女声音色（默认 zh-CN-XiaoxiaoNeural）
  --lang         语音识别语言（默认 zh，英文用 en）
  --skip-frames  跳帧数（默认 2，即每3帧重绘1帧，提速用）
  --strength     重绘强度 0.0~1.0（默认 0.75，越高变化越大）
  --no-face-swap 跳过第2步换脸精修（只做全身重绘）
"""

import argparse
import os
import sys
import shutil
import logging
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def process(
    input_video: str,
    female_face: str,
    output_path: str,
    tts_voice: str = 'zh-CN-XiaoxiaoNeural',
    asr_lang: str = 'zh',
    skip_frames: int = 2,
    strength: float = 0.75,
    no_face_swap: bool = False,
    keep_temp: bool = False,
) -> bool:

    # 使用国内镜像下载模型
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

    from src.utils import ensure_dir, get_timestamp
    work_dir = Path('data') / f'temp_{get_timestamp()}'
    ensure_dir(work_dir)

    logger.info("=" * 60)
    logger.info("  完整性别转换: 全身重绘 + 换脸 + 换声音  ")
    logger.info("=" * 60)
    logger.info(f"输入视频  : {input_video}")
    logger.info(f"女性人脸  : {female_face}")
    logger.info(f"输出路径  : {output_path}")
    logger.info(f"跳帧模式  : skip_frames={skip_frames}")
    logger.info(f"重绘强度  : {strength}")
    logger.info("=" * 60)

    try:
        # ------------------------------------------------------------
        # 步骤 1: ControlNet + SD 全身重绘
        # ------------------------------------------------------------
        logger.info("\n[步骤 1/4] 全身重绘（ControlNet + Stable Diffusion）...")
        from src.body_swapper import BodySwapper

        body_swapper = BodySwapper(strength=strength, device='cuda')
        body_swapped = str(work_dir / 'body_swapped_noaudio.mp4')

        ok = body_swapper.process_video(
            video_path=input_video,
            output_path=body_swapped,
            skip_frames=skip_frames,
        )
        if not ok:
            logger.error("[FAIL] 全身重绘失败，流程中断")
            return False
        logger.info(f"[OK] 步骤 1 完成 -> {body_swapped}")

        # ------------------------------------------------------------
        # 步骤 2: InsightFace 换脸精修（可选）
        # ------------------------------------------------------------
        if no_face_swap:
            face_swapped = body_swapped
            logger.info("\n[步骤 2/4] 换脸精修已跳过（--no-face-swap）")
        else:
            logger.info("\n[步骤 2/4] 换脸精修（InsightFace）...")
            from src.face_swapper import FaceSwapper

            swapper = FaceSwapper()
            face_swapped = str(work_dir / 'face_swapped_noaudio.mp4')
            ok = swapper.swap_male_faces_in_video(
                source_image_path=female_face,
                video_path=body_swapped,
                output_path=face_swapped,
            )
            if not ok:
                logger.warning("换脸精修失败，使用全身重绘结果继续")
                face_swapped = body_swapped
            else:
                logger.info(f"[OK] 步骤 2 完成 -> {face_swapped}")

        # ------------------------------------------------------------
        # 步骤 3: 语音识别 + 女声合成
        # ------------------------------------------------------------
        logger.info("\n[步骤 3/4] 语音识别 + 女声生成...")
        from src.audio_processor import ASRProcessor, TTSProcessor

        asr = ASRProcessor(model_size='small', device='cuda')
        transcript = asr.transcribe_video(input_video, language=asr_lang)

        if not transcript.strip():
            logger.warning("未识别到语音内容，将使用静音音频")
            female_audio = str(work_dir / 'silence.mp3')
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

        logger.info(f"[OK] 步骤 3 完成 -> {female_audio}")

        # ------------------------------------------------------------
        # 步骤 4: 合并视频 + 音频
        # ------------------------------------------------------------
        logger.info("\n[步骤 4/4] 合并视频画面与女声音频...")

        out_dir = Path(output_path).parent
        if str(out_dir) != '.':
            ensure_dir(out_dir)

        from src.video_composer import VideoComposer
        composer = VideoComposer()
        ok = composer.compose_video_with_audio(
            video_path=face_swapped,
            audio_path=female_audio,
            output_path=output_path,
            overwrite=True,
        )
        if not ok:
            logger.error("[FAIL] 视频合成失败")
            return False

        logger.info(f"[OK] 步骤 4 完成 -> {output_path}")
        logger.info("\n" + "=" * 60)
        logger.info(f"  处理完成！输出文件: {output_path}  ")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.exception(f"处理过程中发生错误: {e}")
        logger.info(f"临时文件保留在: {work_dir}")
        return False

    finally:
        if not keep_temp and work_dir.exists():
            if Path(output_path).exists():
                shutil.rmtree(work_dir, ignore_errors=True)
                logger.info(f"临时文件已清理: {work_dir}")
            else:
                logger.info(f"输出文件未生成，临时文件保留在: {work_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='完整男转女: 全身重绘 + 换脸 + 换声',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--input',       required=True, help='输入视频路径')
    parser.add_argument('--face',        required=True, help='女性人脸图片路径')
    parser.add_argument('--output',      default='output_full.mp4', help='输出视频路径')
    parser.add_argument('--voice',       default='zh-CN-XiaoxiaoNeural', help='TTS 女声音色')
    parser.add_argument('--lang',        default='zh', help='ASR 语言代码')
    parser.add_argument('--skip-frames', type=int, default=2,
                        help='跳帧数（0=每帧重绘，2=每3帧重绘1帧，默认2）')
    parser.add_argument('--strength',    type=float, default=0.75,
                        help='重绘强度 0.0~1.0（默认0.75）')
    parser.add_argument('--no-face-swap', action='store_true',
                        help='跳过换脸精修步骤')
    parser.add_argument('--keep-temp',   action='store_true',
                        help='保留临时文件（调试用）')

    args = parser.parse_args()

    for path, name in [(args.input, '输入视频'), (args.face, '女性人脸图片')]:
        if not os.path.exists(path):
            print(f"[FAIL] {name}不存在: {path}")
            sys.exit(1)

    success = process(
        input_video=args.input,
        female_face=args.face,
        output_path=args.output,
        tts_voice=args.voice,
        asr_lang=args.lang,
        skip_frames=args.skip_frames,
        strength=args.strength,
        no_face_swap=args.no_face_swap,
        keep_temp=args.keep_temp,
    )
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
