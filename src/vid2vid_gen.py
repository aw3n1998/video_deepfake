"""
Wan2.1 视频内容生成管线 (Video-to-Video Content Generation)

与 vid2vid.py 的区别：
- vid2vid.py: SD1.5 img2img — 只能改风格/色调，不能改内容
- vid2vid_gen.py: Wan2.1-14B — 可以根据提示词改变动作、物体、场景

技术栈：
- Wan2.1-14B Image-to-Video (diffusers)
- 分段处理: 长视频切分 → 逐段生成 → 段间平滑拼接
- RIFE 帧插值: 平滑段间过渡（可选）

硬件需求：
- 最低 80GB VRAM (A100/H100)
- 支持 multi-GPU (device_map="auto")
"""

import cv2
import gc
import logging
import os
import subprocess
import numpy as np
import torch
from dataclasses import dataclass
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_VIDEO_DURATION = 300  # 5 分钟
DEFAULT_SEGMENT_SECONDS = 20


@dataclass
class VideoSegment:
    """一段待处理的视频片段"""
    frames: List[np.ndarray]
    start_idx: int
    end_idx: int
    first_frame: np.ndarray
    last_frame: np.ndarray


class Wan2VidPipeline:
    """
    基于 Wan2.1 的视频内容生成管线。

    工作流程：
    1. 读取原视频 → 切分为 ≤20秒 的片段
    2. 每段：取首帧作为引导 + 提示词 → Wan2.1 生成新内容
    3. 段间用 RIFE 或交叉淡化平滑过渡
    4. 拼接 + 合并原始音频 → 输出
    """

    def __init__(
        self,
        model_name: str = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        device: str = "cuda",
        device_map: Optional[str] = None,
        dtype: str = "float16",
    ):
        self.model_name = model_name
        self.device = device
        self.device_map = device_map
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self.pipe = None
        self._rife_model = None

    # ════════════════════════════════════════════════════════════
    # 模型初始化
    # ════════════════════════════════════════════════════════════

    def _init_models(self):
        """延迟加载 Wan2.1 管线"""
        from diffusers import WanImageToVideoPipeline
        from diffusers.utils import export_to_video

        logger.info(f"加载 Wan2.1 模型: {self.model_name}")
        logger.info("这可能需要几分钟（首次加载需下载 ~28GB 权重）...")

        load_kwargs = dict(torch_dtype=self.dtype)
        if self.device_map:
            load_kwargs["device_map"] = self.device_map

        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.model_name, **load_kwargs
        )

        if not self.device_map:
            self.pipe.enable_model_cpu_offload()

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers 显存优化已启用")
        except Exception:
            logger.info("xformers 不可用，使用默认注意力机制")

        logger.info("Wan2.1 管线初始化完成")

    def _init_rife(self) -> bool:
        """尝试加载 RIFE 帧插值模型"""
        if self._rife_model is not None:
            return True
        try:
            from rife_ncnn_vulkan import Rife
            self._rife_model = Rife(model=4, gpuid=0)
            logger.info("RIFE 帧插值已加载")
            return True
        except ImportError:
            logger.info("RIFE 不可用，段间过渡将使用交叉淡化")
            return False

    # ════════════════════════════════════════════════════════════
    # 视频分段
    # ════════════════════════════════════════════════════════════

    @staticmethod
    def _segment_video(
        frames: List[np.ndarray],
        fps: float,
        max_segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    ) -> List[VideoSegment]:
        """
        将视频帧序列切分为多个片段。

        相邻片段共享边界帧以保证过渡连贯。
        """
        max_frames = max(int(fps * max_segment_seconds), 24)
        total = len(frames)
        segments = []

        start = 0
        while start < total:
            end = min(start + max_frames, total)
            seg_frames = frames[start:end]
            segments.append(VideoSegment(
                frames=seg_frames,
                start_idx=start,
                end_idx=end,
                first_frame=seg_frames[0].copy(),
                last_frame=seg_frames[-1].copy(),
            ))
            start = end

        logger.info(f"视频分段: {len(segments)} 段 | 每段最长 {max_segment_seconds}秒")
        return segments

    # ════════════════════════════════════════════════════════════
    # 单段处理
    # ════════════════════════════════════════════════════════════

    def _process_segment(
        self,
        segment: VideoSegment,
        prompt: str,
        negative_prompt: str,
        num_steps: int,
        guidance_scale: float,
        resolution: int,
        seed: int,
        prev_last_generated: Optional[Image.Image] = None,
    ) -> List[np.ndarray]:
        """
        用 Wan2.1 处理单个视频片段。

        使用首帧（或上一段的最后一帧）作为图像引导，
        结合提示词生成新的视频内容。
        """
        # 选择引导图：优先用上一段末帧（保证段间连贯）
        if prev_last_generated is not None:
            guide_image = prev_last_generated
        else:
            guide_image = Image.fromarray(
                cv2.cvtColor(segment.first_frame, cv2.COLOR_BGR2RGB)
            )

        # 缩放到目标分辨率
        orig_h, orig_w = segment.first_frame.shape[:2]
        scale = resolution / max(orig_w, orig_h)
        proc_w = max(int(orig_w * scale) // 8 * 8, 512)
        proc_h = max(int(orig_h * scale) // 8 * 8, 512)
        guide_image = guide_image.resize((proc_w, proc_h), Image.LANCZOS)

        # Wan2.1 生成的帧数 (与原始片段对齐)
        num_frames = len(segment.frames)
        # Wan2.1 对帧数有要求，需为 4 的倍数 +1
        num_frames = max(((num_frames - 1) // 4) * 4 + 1, 9)

        generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info(
            f"  生成 {num_frames} 帧 | 分辨率 {proc_w}x{proc_h} | "
            f"steps={num_steps} | cfg={guidance_scale}"
        )

        output = self.pipe(
            image=guide_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        # 转换回 BGR numpy 数组并缩放回原分辨率
        generated_frames = []
        for frame_pil in output.frames[0]:
            if not isinstance(frame_pil, Image.Image):
                frame_pil = Image.fromarray(frame_pil)
            frame_pil = frame_pil.resize((orig_w, orig_h), Image.LANCZOS)
            frame_bgr = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            generated_frames.append(frame_bgr)

        # 如果生成帧数与原始不匹配，截断或重复最后一帧
        target_len = len(segment.frames)
        if len(generated_frames) > target_len:
            generated_frames = generated_frames[:target_len]
        while len(generated_frames) < target_len:
            generated_frames.append(generated_frames[-1].copy())

        return generated_frames

    # ════════════════════════════════════════════════════════════
    # 段间平滑
    # ════════════════════════════════════════════════════════════

    def _smooth_junctions(
        self,
        all_frames: List[np.ndarray],
        junction_indices: List[int],
        blend_window: int = 8,
    ) -> List[np.ndarray]:
        """
        平滑段间过渡。

        在每个分段边界做交叉淡化（或 RIFE 插值）以消除跳变。
        """
        if not junction_indices:
            return all_frames

        result = list(all_frames)
        has_rife = self._init_rife()

        for junction in junction_indices:
            half = blend_window // 2
            start = max(0, junction - half)
            end = min(len(result), junction + half)

            if has_rife and self._rife_model is not None:
                # RIFE: 在边界处插入光流插值帧
                try:
                    before = result[max(0, junction - 1)]
                    after = result[min(len(result) - 1, junction)]
                    before_pil = Image.fromarray(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
                    after_pil = Image.fromarray(cv2.cvtColor(after, cv2.COLOR_BGR2RGB))
                    mid_pil = self._rife_model.process(before_pil, after_pil)
                    mid_bgr = cv2.cvtColor(np.array(mid_pil), cv2.COLOR_RGB2BGR)
                    result[junction] = mid_bgr
                except Exception as e:
                    logger.warning(f"RIFE 插值失败，使用交叉淡化: {e}")
                    self._crossfade(result, start, end)
            else:
                self._crossfade(result, start, end)

        return result

    @staticmethod
    def _crossfade(frames: List[np.ndarray], start: int, end: int):
        """在 [start, end) 区间做交叉淡化"""
        length = end - start
        if length < 2:
            return

        anchor_before = frames[start].copy()
        anchor_after = frames[end - 1].copy()

        for i in range(start, end):
            alpha = (i - start) / (length - 1)
            frames[i] = cv2.addWeighted(
                anchor_after, alpha, anchor_before, 1.0 - alpha, 0
            )

    # ════════════════════════════════════════════════════════════
    # 主处理流程
    # ════════════════════════════════════════════════════════════

    def process_video(
        self,
        video_path: str,
        output_path: str,
        prompt: str,
        negative_prompt: str = (
            "low quality, blurry, deformed, ugly, bad anatomy, "
            "disfigured, watermark, text, static, motionless"
        ),
        num_steps: int = 30,
        guidance_scale: float = 5.0,
        max_segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
        resolution: int = 720,
        junction_blend_frames: int = 8,
        seed: int = 42,
    ) -> bool:
        """
        基于 Wan2.1 的视频内容生成。

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            prompt: 正向提示词（描述想要生成的内容/动作）
            negative_prompt: 负向提示词
            num_steps: 推理步数 (越高质量越好，速度越慢)
            guidance_scale: 提示词引导强度
            max_segment_seconds: 每段最大时长（秒）
            resolution: 处理分辨率 (480/720)
            junction_blend_frames: 段间过渡帧数
            seed: 随机种子

        Returns:
            成功返回 True
        """
        # 延迟初始化
        if self.pipe is None:
            self._init_models()

        if not os.path.isfile(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            return False

        # 读取视频
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            logger.error("视频为空，无法处理")
            return False

        total = len(frames)
        duration = total / fps if fps > 0 else 0
        logger.info(
            f"视频信息: {total} 帧 | {fps:.1f}fps | "
            f"{orig_w}x{orig_h} | {duration:.1f}秒"
        )

        if duration > MAX_VIDEO_DURATION:
            logger.error(
                f"视频时长 {duration:.0f}秒 超过上限 {MAX_VIDEO_DURATION}秒"
            )
            return False

        # 分段
        segments = self._segment_video(frames, fps, max_segment_seconds)

        # 逐段处理
        all_generated: List[np.ndarray] = []
        junction_indices: List[int] = []
        prev_last_pil: Optional[Image.Image] = None

        for seg_idx, segment in enumerate(segments):
            logger.info(
                f"处理分段 {seg_idx + 1}/{len(segments)} "
                f"(帧 {segment.start_idx}-{segment.end_idx})"
            )

            try:
                gen_frames = self._process_segment(
                    segment=segment,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    resolution=resolution,
                    seed=seed + seg_idx,
                    prev_last_generated=prev_last_pil,
                )

                # 记录段间边界位置
                if all_generated:
                    junction_indices.append(len(all_generated))

                all_generated.extend(gen_frames)

                # 保存最后一帧作为下一段的引导
                last_bgr = gen_frames[-1]
                prev_last_pil = Image.fromarray(
                    cv2.cvtColor(last_bgr, cv2.COLOR_BGR2RGB)
                )

            except Exception as e:
                logger.warning(
                    f"分段 {seg_idx + 1} 生成失败: {e}，使用原始帧"
                )
                if all_generated:
                    junction_indices.append(len(all_generated))
                all_generated.extend(segment.frames)
                prev_last_pil = None

            # 段间释放显存
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

        # 段间平滑
        if junction_indices:
            logger.info(f"平滑 {len(junction_indices)} 个段间过渡...")
            all_generated = self._smooth_junctions(
                all_generated, junction_indices, junction_blend_frames
            )

        # 写入临时视频
        logger.info("写入视频帧...")
        temp_video = output_path.replace(".mp4", "_temp_noaudio.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_video, fourcc, fps, (orig_w, orig_h))
        for f in all_generated:
            writer.write(f)
        writer.release()

        # 合并原始音频
        logger.info("合并原始音频...")
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg",
                    "-i", temp_video,
                    "-i", video_path,
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-map", "0:v:0",
                    "-map", "1:a:0?",
                    "-shortest",
                    "-y", output_path,
                ],
                check=True,
                capture_output=True,
            )
            if os.path.exists(temp_video):
                os.remove(temp_video)
            logger.info(f"视频生成完成: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"FFmpeg 音频合并失败: {e}")
            if os.path.exists(temp_video):
                os.rename(temp_video, output_path)
                logger.info(f"已保存无音频版本: {output_path}")

        return True
