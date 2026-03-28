"""
全身重绘模块 v3 - AnimateDiff + ControlNet
- 16帧一组联合生成，帧间时序注意力机制保证流畅无闪烁
- 多线程 I/O 流水线（读帧/写帧与 GPU 推理并行）
- 块间重叠线性融合消除块接缝
"""

import cv2
import gc
import io
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

CHUNK_SIZE = 16   # AnimateDiff 标准处理帧数
OVERLAP    = 4    # 相邻块重叠帧数，用于平滑过渡


class BodySwapper:
    DEFAULT_PROMPT = (
        "beautiful young woman, female, long hair, feminine appearance, "
        "photorealistic, high quality, 8k uhd, professional photography, "
        "natural lighting, detailed face, consistent appearance"
    )
    DEFAULT_NEG = (
        "male, man, boy, masculine, beard, mustache, stubble, "
        "ugly, deformed, blurry, low quality, watermark, nsfw, "
        "flickering, inconsistent, temporal artifacts"
    )

    def __init__(
        self,
        sd_model: str = "SG161222/Realistic_Vision_V5.1_noVAE",
        motion_adapter: str = "guoyww/animatediff-motion-adapter-v1-5-2",
        controlnet_model: str = "lllyasviel/control_v11p_sd15_openpose",
        device: str = "cuda",
        strength: float = 0.85,
        guidance_scale: float = 7.5,
        batch_size: int = 4,   # 保留兼容性，AnimateDiff 此参数无效
    ):
        self.device = device
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.sd_model = sd_model
        self.motion_adapter = motion_adapter
        self.controlnet_model = controlnet_model

        self.pipe = None
        self.pose_detector = None
        self.face_analyser = None
        self._rembg_remove = None

        self._init_models()

    # ────────────────────────────────────────────────────────
    # 模型初始化
    # ────────────────────────────────────────────────────────

    def _init_models(self):
        self._init_face_analyser()
        self._init_pose_detector()
        self._init_segmentation()
        self._init_animatediff_pipeline()

    def _init_face_analyser(self):
        try:
            import insightface
            providers = (
                ['CUDAExecutionProvider', 'CPUExecutionProvider']
                if self.device == 'cuda' else ['CPUExecutionProvider']
            )
            self.face_analyser = insightface.app.FaceAnalysis(
                name='buffalo_l', providers=providers
            )
            self.face_analyser.prepare(
                ctx_id=0 if self.device == 'cuda' else -1,
                det_size=(640, 640)
            )
            logger.info("[OK] InsightFace 加载成功")
        except Exception as e:
            logger.warning(f"[警告] InsightFace 未安装或加载失败，跳过人脸检测: {e}")
            logger.warning("  提示: pip install insightface")
            self.face_analyser = None

    def _init_pose_detector(self):
        try:
            from controlnet_aux import OpenposeDetector
            self.pose_detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
            logger.info("[OK] OpenPose 检测器加载成功")
        except Exception as e:
            logger.warning(f"[警告] controlnet_aux 未安装或加载失败，跳过姿态检测: {e}")
            logger.warning("  提示: pip install controlnet-aux")
            self.pose_detector = None

    def _init_segmentation(self):
        try:
            from rembg import remove
            self._rembg_remove = remove
            logger.info("[OK] rembg 人体分割加载成功")
        except Exception as e:
            logger.warning(f"rembg 不可用，使用全帧掩码: {e}")

    def _init_animatediff_pipeline(self):
        try:
         self._init_animatediff_pipeline_inner()
        except Exception as e:
            logger.warning(f"[警告] AnimateDiff 流水线加载失败，全身重绘将跳过: {e}")
            logger.warning("  提示: pip install diffusers transformers accelerate")
            self.pipe = None

    def _init_animatediff_pipeline_inner(self):
        from diffusers import (
            AnimateDiffControlNetPipeline,
            ControlNetModel,
            MotionAdapter,
            DDIMScheduler,
        )

        # CPU 不支持 float16，自动降级为 float32
        dtype = torch.float16 if self.device == 'cuda' else torch.float32

        logger.info(f"加载 MotionAdapter: {self.motion_adapter} ...")
        adapter = MotionAdapter.from_pretrained(
            self.motion_adapter, torch_dtype=dtype
        )

        logger.info(f"加载 ControlNet: {self.controlnet_model} ...")
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model, torch_dtype=dtype
        )

        logger.info(f"加载 Stable Diffusion: {self.sd_model} ...")
        self.pipe = AnimateDiffControlNetPipeline.from_pretrained(
            self.sd_model,
            motion_adapter=adapter,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(self.device)

        self.pipe.scheduler = DDIMScheduler.from_pretrained(
            self.sd_model,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )

        self.pipe.enable_model_cpu_offload()
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            logger.info("[OK] xformers 显存优化已启用")
        except Exception:
            logger.info("xformers 不可用，使用默认注意力")

        logger.info("[OK] AnimateDiff + ControlNet pipeline 加载完成")

    # ────────────────────────────────────────────────────────
    # CPU 预处理（可多线程并行）
    # ────────────────────────────────────────────────────────

    def has_male(self, frames: List[np.ndarray]) -> bool:
        """在帧列表中采样检测是否含男性（每4帧检测一次）"""
        for frame in frames[::4]:
            try:
                faces = self.face_analyser.get(frame)
                if any(getattr(f, 'gender', 0) == 1 for f in faces):
                    return True
            except Exception:
                pass
        return False

    def _extract_one_pose(self, frame: np.ndarray) -> Image.Image:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.pose_detector(pil)
        return result if isinstance(result, Image.Image) else Image.fromarray(result)

    def extract_poses(self, frames: List[np.ndarray]) -> List[Image.Image]:
        """多线程并行提取姿态图（CPU密集，与GPU互不干扰）"""
        with ThreadPoolExecutor(max_workers=min(4, len(frames))) as pool:
            return list(pool.map(self._extract_one_pose, frames))

    def segment_person(self, frame: np.ndarray) -> np.ndarray:
        if self._rembg_remove:
            buf = io.BytesIO()
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(buf, format='PNG')
            out = self._rembg_remove(buf.getvalue())
            alpha = np.array(Image.open(io.BytesIO(out)).convert('RGBA'))[:, :, 3]
            mask = (alpha > 128).astype(np.uint8) * 255
            return cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
        return np.full(frame.shape[:2], 255, dtype=np.uint8)

    # ────────────────────────────────────────────────────────
    # AnimateDiff GPU 推理
    # ────────────────────────────────────────────────────────

    def generate_chunk(
        self,
        frames: List[np.ndarray],
        pose_images: List[Image.Image],
        seed: int = 42,
    ) -> List[np.ndarray]:
        """
        AnimateDiff 处理一个 chunk，输出时序一致的女性外观帧序列。
        seed 固定保证同一 chunk 每次结果一致。
        """
        h, w = frames[0].shape[:2]
        gw = max((w // 8) * 8, 512)
        gh = max((h // 8) * 8, 512)

        resized_poses = [p.resize((gw, gh)) for p in pose_images]

        with torch.autocast(self.device):
            output = self.pipe(
                prompt=self.DEFAULT_PROMPT,
                negative_prompt=self.DEFAULT_NEG,
                num_frames=len(frames),
                num_inference_steps=20,
                guidance_scale=self.guidance_scale,
                conditioning_frames=resized_poses,
                generator=torch.Generator(device=self.device).manual_seed(seed),
            )

        frames_out = (output.frames[0]
                      if isinstance(output.frames[0], list)
                      else output.frames)

        return [
            cv2.cvtColor(np.array(pil.resize((w, h))), cv2.COLOR_RGB2BGR)
            for pil in frames_out
        ]

    def blend_overlap(
        self,
        prev: List[np.ndarray],
        curr: List[np.ndarray],
    ) -> List[np.ndarray]:
        """对重叠区域做线性 alpha 渐变融合，消除块间接缝"""
        result = []
        for i in range(OVERLAP):
            alpha = (i + 1) / (OVERLAP + 1)
            frame = cv2.addWeighted(
                prev[len(prev) - OVERLAP + i], 1 - alpha,
                curr[i], alpha, 0
            )
            result.append(frame)
        return result

    def blend_with_bg(
        self,
        original: np.ndarray,
        generated: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """将生成的人物通过分割掩码融合回原始背景"""
        m = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 0) / 255.0
        m3 = m[:, :, np.newaxis]
        return (generated.astype(np.float32) * m3 +
                original.astype(np.float32) * (1 - m3)).astype(np.uint8)

    # ────────────────────────────────────────────────────────
    # 视频处理主流程
    # ────────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        output_path: str,
        skip_frames: int = 0,   # AnimateDiff 不建议跳帧，建议保持0
    ) -> bool:
        """
        处理流程：
          Thread-Reader  → 读帧入队列
          Main (GPU)     → AnimateDiff 分块推理（含姿态提取并行）
          Thread-Writer  → 写帧到文件

        每 CHUNK_SIZE=16 帧为一组，相邻组重叠 OVERLAP=4 帧线性融合。
        """
        if self.pipe is None:
            logger.error("[FAIL] AnimateDiff 流水线未加载，无法进行全身重绘")
            logger.error("  请安装: pip install diffusers transformers accelerate")
            logger.error("  或在界面勾选【本地测试模式】跳过全身重绘")
            return False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {video_path}")
            return False

        fps   = cap.get(cv2.CAP_PROP_FPS)
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out_writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
        )

        logger.info(f"视频: {total}帧 | {fps:.1f}fps | {w}x{h}")
        logger.info(f"AnimateDiff: chunk={CHUNK_SIZE}, overlap={OVERLAP}, "
                    f"stride={CHUNK_SIZE - OVERLAP}")

        # ── 多线程读帧 ─────────────────────────────────────
        read_q: queue.Queue = queue.Queue(maxsize=128)

        def reader_worker():
            while True:
                ret, frame = cap.read()
                if not ret:
                    read_q.put(None)
                    break
                read_q.put(frame)

        reader_t = threading.Thread(target=reader_worker, daemon=True)
        reader_t.start()

        all_frames: List[np.ndarray] = []
        while True:
            f = read_q.get()
            if f is None:
                break
            all_frames.append(f)
        cap.release()
        reader_t.join()
        logger.info(f"读入 {len(all_frames)} 帧完成")

        # ── AnimateDiff 分块处理 ────────────────────────────
        processed: List[Optional[np.ndarray]] = [None] * len(all_frames)
        stride = CHUNK_SIZE - OVERLAP
        prev_generated: Optional[List[np.ndarray]] = None
        chunk_idx = 0
        i = 0

        # CPU姿态提取与GPU推理预取：提前准备下一块姿态
        pose_executor = ThreadPoolExecutor(max_workers=4)
        next_pose_future = None

        while i < len(all_frames):
            end = min(i + CHUNK_SIZE, len(all_frames))
            chunk_raw = all_frames[i:end]
            pad = CHUNK_SIZE - len(chunk_raw)
            chunk = chunk_raw + [chunk_raw[-1]] * pad

            chunk_idx += 1
            pct = i / len(all_frames) * 100
            logger.info(f"[Chunk {chunk_idx}] 帧 {i}~{end - 1} ({pct:.1f}%)"
                        + (f" +{pad}帧补齐" if pad else ""))

            if self.has_male(chunk):
                # 获取姿态（若有预取则直接取结果）
                if next_pose_future is not None:
                    poses = next_pose_future.result()
                else:
                    poses = self.extract_poses(chunk)

                # 预取下一块姿态（与当前GPU推理并行）
                next_i = i + stride
                if next_i < len(all_frames):
                    next_end = min(next_i + CHUNK_SIZE, len(all_frames))
                    next_chunk = all_frames[next_i:next_end]
                    npad = CHUNK_SIZE - len(next_chunk)
                    next_chunk = next_chunk + [next_chunk[-1]] * npad
                    next_pose_future = pose_executor.submit(
                        self.extract_poses, next_chunk
                    )
                else:
                    next_pose_future = None

                # AnimateDiff GPU 推理
                generated = self.generate_chunk(chunk, poses, seed=chunk_idx)

                # 重叠区域融合（修正前一块末尾）
                if prev_generated is not None:
                    blended = self.blend_overlap(prev_generated, generated)
                    for j, bf in enumerate(blended):
                        idx = i - OVERLAP + j
                        if 0 <= idx < len(all_frames):
                            mask = self.segment_person(all_frames[idx])
                            processed[idx] = self.blend_with_bg(
                                all_frames[idx], bf, mask
                            )

                # 写入当前块（非重叠部分）
                for j in range(len(chunk_raw)):
                    idx = i + j
                    if processed[idx] is None:
                        mask = self.segment_person(all_frames[idx])
                        processed[idx] = self.blend_with_bg(
                            all_frames[idx], generated[j], mask
                        )

                prev_generated = generated

            else:
                # 无男性帧，直接保留原帧
                for j in range(len(chunk_raw)):
                    idx = i + j
                    if processed[idx] is None:
                        processed[idx] = all_frames[idx]
                prev_generated = None
                next_pose_future = None

            gc.collect()
            torch.cuda.empty_cache()
            i += stride

        pose_executor.shutdown(wait=False)

        # ── 多线程写帧 ─────────────────────────────────────
        write_q: queue.Queue = queue.Queue(maxsize=128)

        def writer_worker():
            while True:
                item = write_q.get()
                if item is None:
                    break
                out_writer.write(item)

        writer_t = threading.Thread(target=writer_worker, daemon=True)
        writer_t.start()

        blank = np.zeros((h, w, 3), dtype=np.uint8)
        for frame in processed:
            write_q.put(frame if frame is not None else blank)

        write_q.put(None)
        writer_t.join()
        out_writer.release()

        logger.info(f"[OK] AnimateDiff 视频处理完成: {output_path}")
        return True
