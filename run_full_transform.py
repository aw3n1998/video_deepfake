#!/usr/bin/env python3
"""
完整转换：通用 AI 视频高清重绘（AnimateDiff 全身重绘 + 换脸精修 + 声音处理 + BGM 保留）
多线程版：并行处理音频流水线与视频生成，极致提升 5090 运行效率。

处理流程:
  1. 画面端：AnimateDiff + ControlNet 联合驱动，16 帧一组块状渲染，块间重叠融合确保无闪烁。
  2. 精修端：使用 InsightFace 进行高保真人脸重绘（可选性别过滤或全替换）。
  3. 音频端：音频分离 -> 目标音色参数转换/变调 -> 与原 BGM 重新混音。

用法:
  python run_full_transform.py --input 视频.mp4 --face 目标人脸.png --output 输出.mp4

参数:
  --input        输入视频（必填）
  --face         目标人脸图片（必填）
  --output       输出路径（默认 output_full.mp4）
  --strength     重绘强度（默认 0.85，越高变化越大）
  --semitones    声音变调半音数（默认 5）
  --no-face-swap 跳过换脸精修
  --keep-temp    保留临时文件
  --segments     视频分段数（默认1，内存不足时设4）
"""

import argparse
import os
import sys
import shutil
import logging
import subprocess
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# 音频流水线（在独立线程中运行，与视频处理并行）
# ════════════════════════════════════════════════════════════

def _audio_pipeline(input_video: str, work_dir: Path, semitones: int,
                    result: dict) -> None:
    """
    独立线程：人声/BGM分离 → 变调 → 混合
    结果写入 result['audio_path']，失败写入 result['error']
    """
    try:
        import numpy as np
        import librosa
        import soundfile as sf
        from pydub import AudioSegment

        raw_wav    = str(work_dir / 'raw_audio.wav')
        vocals_wav = None
        bgm_wav    = None

        # ── 1. 提取音频 ────────────────────────────────────
        logger.info("[Audio] 提取音频...")
        subprocess.run(
            ['ffmpeg', '-i', input_video, '-vn', '-ar', '44100',
             '-ac', '2', '-y', raw_wav],
            capture_output=True, check=True
        )

        # ── 2. demucs 分离人声 / BGM ───────────────────────
        logger.info("[Audio] demucs 分离人声和BGM（此步约5分钟）...")
        try:
            subprocess.run(
                ['python', '-m', 'demucs', '--two-stems=vocals',
                 '-o', str(work_dir / 'demucs_out'), raw_wav],
                check=True
            )
        except subprocess.CalledProcessError as demucs_err:
            logger.warning(f"[Audio] demucs 失败，跳过音频分离，直接使用原始音频: {demucs_err}")
            logger.warning("[Audio] 提示: pip install demucs==4.0.1 可修复此问题")
            result['audio_path'] = raw_wav
            return
        stem      = Path(raw_wav).stem
        demucs_dir = work_dir / 'demucs_out' / 'htdemucs' / stem
        vocals_wav = str(demucs_dir / 'vocals.wav')
        bgm_wav    = str(demucs_dir / 'no_vocals.wav')
        logger.info(f"[Audio] 人声: {vocals_wav}")
        logger.info(f"[Audio] BGM : {bgm_wav}")

        # ── 3. 检测声段 + 变调 ────────────────────────
        logger.info("[Audio] 检测语音片段并变调...")
        from inaSpeechSegmenter import Segmenter
        segments = Segmenter()(vocals_wav)
        
        y, sr = librosa.load(vocals_wav, sr=None, mono=False)
        if y.ndim == 1:
            y = np.array([y, y])
        result_audio = y.copy()

        for label, start, end in segments:
            if label == 'noEnergy':
                continue
            s, e = int(start * sr), int(end * sr)
            logger.info(f"[Audio]   变调: {start:.1f}s ~ {end:.1f}s")
            for ch in range(result_audio.shape[0]):
                result_audio[ch, s:e] = librosa.effects.pitch_shift(
                    y[ch, s:e], sr=sr, n_steps=semitones
                )

        converted_wav = str(work_dir / 'converted_vocals.wav')
        sf.write(converted_wav, result_audio.T, sr)
        logger.info(f"[Audio] 变调完成: {converted_wav}")

        # ── 4. 混合人声 + BGM ──────────────────────────────
        logger.info("[Audio] 混合人声 + BGM...")
        v = AudioSegment.from_wav(converted_wav)
        b = AudioSegment.from_wav(bgm_wav)
        mixed_wav = str(work_dir / 'mixed_audio.wav')
        v.overlay(b).export(mixed_wav, format='wav')
        logger.info(f"[Audio] 混合完成: {mixed_wav}")

        result['audio_path'] = mixed_wav

    except Exception as e:
        logger.exception(f"[Audio] 音频处理失败: {e}")
        result['error'] = str(e)


# ════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════

def process(
    input_video: str,
    target_face: str,
    output_path: str = 'output_full.mp4',
    skip_frames: int = 0,
    batch_size: int = 1,
    strength: float = 0.85,
    semitones: int = 5,
    filter_gender: int = -1,
    no_face_swap: bool = False,
    no_body_swap: bool = False,
    keep_temp: bool = False,
) -> bool:

    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

    from src.utils import ensure_dir, get_timestamp
    work_dir = Path('data') / f'temp_{get_timestamp()}'
    ensure_dir(work_dir)

    logger.info("=" * 60)
    logger.info("  通用 AI 视频重塑（多线程并行版）")
    logger.info(f"  视频={input_video}")
    logger.info(f"  人脸={target_face}")
    logger.info(f"  strength={strength}  semitones={semitones}")
    logger.info("=" * 60)

    try:
        # ── 启动音频线程（与视频处理并行）─────────────────
        audio_result: dict = {}
        audio_thread = threading.Thread(
            target=_audio_pipeline,
            args=(input_video, work_dir, semitones, audio_result),
            daemon=True,
            name='AudioPipeline'
        )
        audio_thread.start()
        logger.info("[Main] 音频处理线程已启动，与视频处理并行运行")

        # ── 步骤 1: 全身重绘 ───────────────────────────────
        if no_body_swap:
            logger.info("\n[步骤 1/3] 全身重绘已跳过（本地测试模式）")
            body_swapped = input_video
        else:
            logger.info("\n[步骤 1/3] 全身重绘（ControlNet + SD）...")
            from src.body_swapper import BodySwapper

            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            body_swapper = BodySwapper(
                strength=strength,
                device=device,
                batch_size=batch_size,
            )
            body_swapped = str(work_dir / 'body_swapped.mp4')
            ok = body_swapper.process_video(
                video_path=input_video,
                output_path=body_swapped,
                skip_frames=skip_frames,
            )
            if not ok:
                logger.error("[FAIL] 全身重绘失败")
                return False
            logger.info(f"[OK] 步骤 1 完成 -> {body_swapped}")

        # ── 步骤 2: 换脸精修 ───────────────────────────────
        if no_face_swap:
            face_swapped = body_swapped
            logger.info("\n[步骤 2/3] 换脸精修已跳过")
        else:
            logger.info("\n[步骤 2/3] 换脸精修（InsightFace）...")
            from src.face_swapper import FaceSwapper
            swapper = FaceSwapper()
            face_swapped = str(work_dir / 'face_swapped.mp4')
            # 使用通用换脸接口，默认全替换
            ok = swapper.swap_faces_by_gender(
                source_image_path=target_face,
                video_path=body_swapped,
                output_path=face_swapped,
                filter_gender=filter_gender
            )
            if not ok:
                logger.warning("换脸失败，使用全身重绘结果继续")
                face_swapped = body_swapped
            else:
                logger.info(f"[OK] 步骤 2 完成 -> {face_swapped}")

        # ── 等待音频线程完成 ───────────────────────────────
        logger.info("\n[步骤 3/3] 等待音频处理完成并合并...")
        audio_thread.join()

        if 'error' in audio_result:
            logger.error(f"音频处理失败: {audio_result['error']}，使用原始音频")
            # 回退：直接从原视频提取音频
            fallback_audio = str(work_dir / 'fallback_audio.aac')
            subprocess.run(
                ['ffmpeg', '-i', input_video, '-vn', '-c:a', 'copy',
                 '-y', fallback_audio],
                capture_output=True
            )
            audio_path = fallback_audio
        else:
            audio_path = audio_result['audio_path']
            logger.info(f"[OK] 音频就绪: {audio_path}")

        # ── 合并视频 + 音频 ────────────────────────────────
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            'ffmpeg', '-i', face_swapped, '-i', audio_path,
            '-c:v', 'copy', '-c:a', 'aac',
            '-map', '0:v:0', '-map', '1:a:0',
            '-shortest', '-y', output_path
        ], check=True)

        logger.info(f"\n[完成] 输出文件: {output_path}")
        return True

    except Exception as e:
        logger.exception(f"处理失败: {e}")
        logger.info(f"临时文件保留在: {work_dir}")
        return False

    finally:
        if not keep_temp and work_dir.exists():
            if Path(output_path).exists():
                shutil.rmtree(work_dir, ignore_errors=True)
            else:
                logger.info(f"输出未生成，临时文件保留: {work_dir}")


# ════════════════════════════════════════════════════════════
# 视频分段并行处理
# ════════════════════════════════════════════════════════════

def _process_segment(seg_input, seg_output, target_face, skip_frames,
                     strength, batch_size, no_face_swap, filter_gender, gpu_id, result, idx):
    """单个视频段的处理函数"""
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        from src.body_swapper import BodySwapper
        from src.face_swapper import FaceSwapper

        body_swapper = BodySwapper(strength=strength, device='cuda', batch_size=batch_size)
        body_tmp = seg_output.replace('.mp4', '_body.mp4')
        body_swapper.process_video(seg_input, body_tmp, skip_frames=skip_frames)

        if no_face_swap:
            shutil.copy(body_tmp, seg_output)
        else:
            swapper = FaceSwapper()
            ok = swapper.swap_faces_by_gender(
                source_image_path=target_face,
                video_path=body_tmp,
                output_path=seg_output,
                filter_gender=filter_gender
            )
            if not ok:
                import shutil
                shutil.copy(body_tmp, seg_output)

        result[idx] = seg_output
        logger.info(f"[Seg-{idx}] 完成: {seg_output}")
    except Exception as e:
        logger.exception(f"[Seg-{idx}] 处理失败: {e}")
        result[idx] = None


def process_with_segments(
    input_video: str,
    female_face: str,
    output_path: str = 'output_full.mp4',
    n_segments: int = 4,
    skip_frames: int = 2,
    strength: float = 0.75,
    batch_size: int = 4,
    semitones: int = 5,
    no_face_swap: bool = False,
    keep_temp: bool = False,
    n_gpus: int = 1,          # 可用GPU数量，单卡填1（分段顺序处理），多卡填实际数量
) -> bool:
    """
    视频分段处理版本：
    - 将视频切成 n_segments 段
    - 单GPU：各段顺序处理，每段结束后立即释放显存
    - 多GPU：各段分配到不同GPU并行处理
    - 音频流水线全程并行
    """
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    from src.utils import ensure_dir, get_timestamp
    work_dir = Path('data') / f'temp_{get_timestamp()}'
    ensure_dir(work_dir)
    seg_dir = work_dir / 'segments'
    ensure_dir(seg_dir)

    logger.info("=" * 60)
    logger.info(f"  分段处理模式: {n_segments} 段 | {n_gpus} GPU")
    logger.info("=" * 60)

    try:
        # ── 启动音频线程 ───────────────────────────────────
        audio_result: dict = {}
        audio_thread = threading.Thread(
            target=_audio_pipeline,
            args=(input_video, work_dir, semitones, audio_result),
            daemon=True, name='AudioPipeline'
        )
        audio_thread.start()
        logger.info("[Main] 音频处理线程已启动")

        # ── 用 FFmpeg 切割视频 ─────────────────────────────
        import cv2
        cap = cv2.VideoCapture(input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        duration = total_frames / fps
        seg_duration = duration / n_segments

        seg_inputs  = []
        seg_outputs = []
        for i in range(n_segments):
            seg_in  = str(seg_dir / f'seg_{i:03d}_in.mp4')
            seg_out = str(seg_dir / f'seg_{i:03d}_out.mp4')
            start = i * seg_duration
            subprocess.run([
                'ffmpeg', '-i', input_video,
                '-ss', str(start), '-t', str(seg_duration),
                '-c', 'copy', '-y', seg_in
            ], capture_output=True, check=True)
            seg_inputs.append(seg_in)
            seg_outputs.append(seg_out)
        logger.info(f"[Main] 视频已切割为 {n_segments} 段，每段约 {seg_duration:.1f}s")

        # ── 处理各段 ───────────────────────────────────────
        seg_results = {}

        if n_gpus > 1:
            # 多GPU：并行处理
            threads = []
            for i, (seg_in, seg_out) in enumerate(zip(seg_inputs, seg_outputs)):
                gpu_id = i % n_gpus
                t = threading.Thread(
                    target=_process_segment,
                    args=(seg_in, seg_out, female_face, skip_frames,
                          strength, batch_size, no_face_swap,
                          -1, gpu_id, seg_results, i),
                    daemon=True
                )
                threads.append(t)
                t.start()
                logger.info(f"[Main] 段 {i} 分配到 GPU {gpu_id}")
            for t in threads:
                t.join()
        else:
            # 单GPU：顺序处理各段，每段结束后释放显存
            for i, (seg_in, seg_out) in enumerate(zip(seg_inputs, seg_outputs)):
                logger.info(f"\n[Main] 处理段 {i+1}/{n_segments}...")
                _process_segment(
                    seg_in, seg_out, female_face, skip_frames,
                    strength, batch_size, no_face_swap,
                    -1, 0, seg_results, i
                )
                # 释放显存
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
                logger.info(f"[Main] 段 {i+1} 完成，显存已释放")

        # ── 合并视频段 ─────────────────────────────────────
        valid_segs = [seg_results.get(i) for i in range(n_segments)
                      if seg_results.get(i)]
        if not valid_segs:
            logger.error("所有段均处理失败")
            return False

        concat_list = str(work_dir / 'concat_list.txt')
        with open(concat_list, 'w') as f:
            for seg in valid_segs:
                f.write(f"file '{os.path.abspath(seg)}'\n")

        merged_video = str(work_dir / 'merged_video.mp4')
        subprocess.run([
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', concat_list, '-c', 'copy', '-y', merged_video
        ], check=True)
        logger.info(f"[Main] 视频段合并完成: {merged_video}")

        # ── 等待音频 + 合并最终输出 ────────────────────────
        logger.info("[Main] 等待音频处理...")
        audio_thread.join()

        audio_path = audio_result.get('audio_path')
        if not audio_path:
            logger.warning("音频处理失败，使用原始音频")
            audio_path = str(work_dir / 'fallback.aac')
            subprocess.run(
                ['ffmpeg', '-i', input_video, '-vn', '-c:a', 'copy', '-y', audio_path],
                capture_output=True
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            'ffmpeg', '-i', merged_video, '-i', audio_path,
            '-c:v', 'copy', '-c:a', 'aac',
            '-map', '0:v:0', '-map', '1:a:0',
            '-shortest', '-y', output_path
        ], check=True)

        logger.info(f"\n[完成] 输出文件: {output_path}")
        return True

    except Exception as e:
        logger.exception(f"处理失败: {e}")
        logger.info(f"临时文件: {work_dir}")
        return False

    finally:
        if not keep_temp and work_dir.exists():
            if Path(output_path).exists():
                shutil.rmtree(work_dir, ignore_errors=True)
            else:
                logger.info(f"临时文件保留: {work_dir}")


def main():
    parser = argparse.ArgumentParser(description='完整男转女（多线程+分段版）')
    parser.add_argument('--input',     required=True, help='输入视频')
    parser.add_argument('--face',      required=True, help='目标人脸图片')
    parser.add_argument('--output',    default='output_full.mp4', help='输出路径')
    parser.add_argument('--strength',  type=float, default=0.85, help='重绘强度 (0.1~1.0)')
    parser.add_argument('--semitones', type=int,   default=5,    help='变调半音')
    parser.add_argument('--gender',    type=int,   default=-1,   help='性别过滤 (1:男, 0:女, -1:全换)')
    parser.add_argument('--no-face-swap', action='store_true', help='跳过换脸')
    parser.add_argument('--batch-size',   type=int,   default=4,
                        help='SD批量推理帧数（96GB显存推荐4~8）')
    parser.add_argument('--segments',     type=int,   default=1,
                        help='视频分段数（1=不分段，4=切成4段顺序处理节省显存）')
    parser.add_argument('--gpus',         type=int,   default=1,
                        help='可用GPU数量（多卡时各段并行）')
    parser.add_argument('--keep-temp',    action='store_true')
    args = parser.parse_args()

    for path, name in [(args.input, '输入视频'), (args.face, '目标人脸图片')]:
        if not os.path.exists(path):
            print(f"[FAIL] {name} 不存在: {path}")
            sys.exit(1)

    if args.segments > 1:
        success = process_with_segments(
            input_video  = args.input,
            female_face  = args.face,
            output_path  = args.output,
            n_segments   = args.segments,
            skip_frames  = args.skip_frames,
            strength     = args.strength,
            batch_size   = args.batch_size,
            semitones    = args.semitones,
            no_face_swap = args.no_face_swap,
            keep_temp    = args.keep_temp,
            n_gpus       = args.gpus,
        )
    else:
        success = process(
            input_video=args.input,
            target_face=args.face,
            output_path=args.output,
            strength=args.strength,
            semitones=args.semitones,
            filter_gender=args.gender,
            no_face_swap=args.no_face_swap,
            keep_temp=args.keep_temp
        )
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
