"""
人物替换模块

功能: 视频中多人场景，选择性替换指定人物的全身外观。
技术: InsightFace(人脸追踪) + SD Inpainting + Depth ControlNet + IP-Adapter

流程:
  1. 首帧检测所有人脸 → 用户选择目标
  2. 逐帧追踪目标人物 (embedding 匹配)
  3. 估算全身遮罩 → SD Inpainting 只重绘该区域
  4. IP-Adapter 注入参考图人物外观
  5. 遮罩融合 + 颜色校正 → 其他人和背景不受影响
"""

import cv2
import gc
import logging
import os
import subprocess
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


class PersonSwapPipeline:
    """
    人物替换管线。

    与全局重绘 (Vid2VidPipeline) 的区别：
    - Vid2Vid: 修改整个画面
    - PersonSwap: 只替换选定的一个人，其他人和背景完全不变
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.face_app = None
        self.pipe = None
        self.depth_estimator = None
        self._ip_adapter_loaded = False

    # ────────────────────────────────────────────────────────
    # 模型初始化
    # ────────────────────────────────────────────────────────

    def _init_face_detector(self):
        """初始化 InsightFace"""
        import insightface

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.face_app = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=providers
        )
        self.face_app.prepare(
            ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640)
        )
        logger.info("✅ InsightFace 初始化完成")

    def _init_inpaint_pipeline(self):
        """初始化 SD Inpainting + ControlNet"""
        from diffusers import (
            StableDiffusionControlNetInpaintPipeline,
            ControlNetModel,
            UniPCMultistepScheduler,
        )

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        logger.info("加载 Depth ControlNet...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=dtype
        )

        logger.info("加载深度估计模型...")
        from transformers import pipeline as hf_pipeline

        self.depth_estimator = hf_pipeline(
            "depth-estimation",
            model="Intel/dpt-large",
            device=0 if self.device == "cuda" else -1,
        )

        logger.info("加载 SD Inpainting 管线...")
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(self.device)

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.enable_model_cpu_offload()
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        logger.info("✅ Inpainting 管线初始化完成")

    def _load_ip_adapter(self):
        """加载 IP-Adapter（将参考图人物外观注入生成）"""
        if self._ip_adapter_loaded:
            return True
        try:
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter_sd15.bin",
            )
            self.pipe.set_ip_adapter_scale(0.6)
            self._ip_adapter_loaded = True
            logger.info("✅ IP-Adapter 已加载")
            return True
        except Exception as e:
            logger.warning(f"IP-Adapter 不可用: {e}")
            return False

    # ────────────────────────────────────────────────────────
    # 人物检测与追踪
    # ────────────────────────────────────────────────────────

    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """
        检测帧中所有人物。

        Returns:
            列表，每个元素:
            {
                'index': int,
                'face_crop': np.ndarray (BGR, 用于UI展示),
                'bbox': [x1, y1, x2, y2],
                'embedding': np.ndarray (512维人脸特征向量),
            }
        """
        if self.face_app is None:
            self._init_face_detector()

        faces = self.face_app.get(frame)
        persons = []

        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]

            # 扩大裁剪区域
            pad = int((x2 - x1) * 0.4)
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w, x2 + pad)
            cy2 = min(h, y2 + pad)
            face_crop = frame[cy1:cy2, cx1:cx2].copy()

            persons.append(
                {
                    "index": i,
                    "face_crop": face_crop,
                    "bbox": bbox,
                    "embedding": face.normed_embedding,
                }
            )

        return persons

    def detect_persons_from_video(self, video_path: str) -> List[np.ndarray]:
        """
        从视频首帧检测人物，返回人脸裁剪图列表（用于 Gradio Gallery 展示）。
        """
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return []

        persons = self.detect_persons(frame)
        # 转 RGB 给 Gradio 展示
        crops = []
        for p in persons:
            rgb_crop = cv2.cvtColor(p["face_crop"], cv2.COLOR_BGR2RGB)
            crops.append(rgb_crop)
        return crops

    def _find_target(
        self,
        frame: np.ndarray,
        target_embedding: np.ndarray,
        threshold: float = 0.35,
    ) -> Optional[np.ndarray]:
        """在帧中找到目标人物的人脸 bbox"""
        faces = self.face_app.get(frame)
        if not faces:
            return None

        best_face = None
        best_sim = -1.0

        for face in faces:
            sim = float(np.dot(target_embedding, face.normed_embedding))
            if sim > best_sim:
                best_sim = sim
                best_face = face

        if best_sim < threshold:
            return None

        return best_face.bbox.astype(int)

    # ────────────────────────────────────────────────────────
    # 全身遮罩估算
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_body_mask(
        frame: np.ndarray,
        face_bbox: np.ndarray,
        body_width_mult: float = 2.5,
        body_height_mult: float = 5.0,
    ) -> np.ndarray:
        """
        从人脸位置估算全身区域遮罩。

        策略:
        - 以人脸为锚点
        - 水平: 人脸宽度 × body_width_mult
        - 垂直: 从人脸顶部往上 0.3 × 脸高，往下 body_height_mult × 脸高
        - 椭圆形遮罩 + 高斯模糊软化边缘
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = face_bbox
        face_w = x2 - x1
        face_h = y2 - y1
        face_cx = (x1 + x2) // 2

        # 身体区域
        body_half_w = int(face_w * body_width_mult / 2)
        body_top = max(0, y1 - int(face_h * 0.5))
        body_bottom = min(h, y2 + int(face_h * body_height_mult))
        body_left = max(0, face_cx - body_half_w)
        body_right = min(w, face_cx + body_half_w)

        # 椭圆遮罩（比矩形更自然）
        mask = np.zeros((h, w), dtype=np.uint8)
        center = ((body_left + body_right) // 2, (body_top + body_bottom) // 2)
        axes = ((body_right - body_left) // 2, (body_bottom - body_top) // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        # 高斯模糊软化边缘
        mask = cv2.GaussianBlur(mask, (61, 61), 25)

        return mask

    # ────────────────────────────────────────────────────────
    # 核心处理
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _color_transfer(generated: np.ndarray, original: np.ndarray) -> np.ndarray:
        """LAB 颜色校正"""
        src_lab = cv2.cvtColor(generated, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
        for ch in range(3):
            s_m, s_s = src_lab[:, :, ch].mean(), src_lab[:, :, ch].std()
            t_m, t_s = tgt_lab[:, :, ch].mean(), tgt_lab[:, :, ch].std()
            if s_s > 1e-6:
                src_lab[:, :, ch] = (src_lab[:, :, ch] - s_m) * (t_s / s_s) + t_m
        return cv2.cvtColor(
            np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
        )

    def process_video(
        self,
        video_path: str,
        output_path: str,
        reference_image_path: str,
        target_person_index: int = 0,
        prompt: str = (
            "same person as reference, natural appearance, consistent lighting, "
            "high quality, detailed, realistic"
        ),
        negative_prompt: str = (
            "low quality, blurry, deformed, ugly, bad anatomy, "
            "different person, wrong face, disfigured"
        ),
        strength: float = 0.65,
        num_steps: int = 25,
        guidance_scale: float = 7.5,
        controlnet_scale: float = 0.7,
        resolution: int = 768,
    ) -> bool:
        """
        替换视频中指定人物。

        Args:
            video_path: 输入视频
            output_path: 输出视频
            reference_image_path: 目标人物参考图
            target_person_index: 要替换的人物索引
            prompt / negative_prompt: 提示词
            strength: 重绘强度 (人物替换推荐 0.55~0.70)
            num_steps: 推理步数
            guidance_scale: CFG 系数
            controlnet_scale: ControlNet 强度
            resolution: 处理分辨率
        """
        # 初始化
        if self.face_app is None:
            self._init_face_detector()
        if self.pipe is None:
            self._init_inpaint_pipeline()

        # 验证
        if not os.path.isfile(video_path):
            logger.error(f"视频不存在: {video_path}")
            return False
        if not os.path.isfile(reference_image_path):
            logger.error(f"参考图不存在: {reference_image_path}")
            return False

        # 参考图
        ref_image = Image.open(reference_image_path).convert("RGB")
        ip_ok = self._load_ip_adapter()

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
            logger.error("视频为空")
            return False

        total = len(frames)
        logger.info(f"视频: {total} 帧 | {fps:.1f}fps | {orig_w}x{orig_h}")

        # 检测目标人物
        persons = self.detect_persons(frames[0])
        if not persons:
            logger.error("首帧未检测到人物")
            return False
        if target_person_index >= len(persons):
            logger.error(
                f"索引 {target_person_index} 超范围 (共 {len(persons)} 人)"
            )
            return False

        target_emb = persons[target_person_index]["embedding"]
        logger.info(
            f"锁定目标人物 #{target_person_index}，共 {len(persons)} 人，开始替换..."
        )

        # 处理分辨率
        scale = resolution / max(orig_w, orig_h)
        proc_w = max(int(orig_w * scale) // 8 * 8, 512)
        proc_h = max(int(orig_h * scale) // 8 * 8, 512)

        # 逐帧处理
        generated_frames = []
        prev_out = None
        gen = torch.Generator(device=self.device).manual_seed(42)

        for i, frame in enumerate(frames):
            # 追踪目标
            target_bbox = self._find_target(frame, target_emb)

            if target_bbox is None:
                # 未检测到 → 保留原帧
                generated_frames.append(frame.copy())
                prev_out = frame.copy()
                continue

            # 全身遮罩
            body_mask = self._estimate_body_mask(frame, target_bbox)

            # 准备输入
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_resized = pil_frame.resize((proc_w, proc_h), Image.LANCZOS)
            mask_resized = cv2.resize(body_mask, (proc_w, proc_h))
            pil_mask = Image.fromarray(mask_resized).convert("L")

            # 深度图
            depth_map = self.depth_estimator(pil_resized)["depth"].convert("RGB")

            # Inpainting
            pipe_kwargs = dict(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pil_resized,
                mask_image=pil_mask,
                control_image=depth_map,
                num_inference_steps=num_steps,
                strength=strength,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_scale,
                generator=gen,
            )
            if ip_ok:
                pipe_kwargs["ip_adapter_image"] = ref_image

            with torch.autocast(self.device):
                result = self.pipe(**pipe_kwargs)

            # 后处理
            gen_pil = result.images[0]
            gen_bgr = cv2.cvtColor(
                np.array(gen_pil.resize((orig_w, orig_h), Image.LANCZOS)),
                cv2.COLOR_RGB2BGR,
            )

            # 遮罩融合：只替换目标人物区域
            mask_f = (body_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
            blended = (
                gen_bgr.astype(np.float32) * mask_f
                + frame.astype(np.float32) * (1.0 - mask_f)
            )
            blended = np.clip(blended, 0, 255).astype(np.uint8)

            # 颜色校正
            blended = self._color_transfer(blended, frame)

            # 时序平滑
            if prev_out is not None:
                blended = cv2.addWeighted(blended, 0.88, prev_out, 0.12, 0)

            generated_frames.append(blended)
            prev_out = blended

            if (i + 1) % 5 == 0 or i == 0 or i == total - 1:
                logger.info(f"  帧 {i+1}/{total} ({100*(i+1)/total:.0f}%)")

            if (i + 1) % 30 == 0:
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        # 写入视频
        temp_video = output_path.replace(".mp4", "_temp_noaudio.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_video, fourcc, fps, (orig_w, orig_h))
        for f in generated_frames:
            writer.write(f)
        writer.release()

        # 合并音频
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg", "-i", temp_video, "-i", video_path,
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k",
                    "-map", "0:v:0", "-map", "1:a:0?",
                    "-shortest", "-y", output_path,
                ],
                check=True,
                capture_output=True,
            )
            if os.path.exists(temp_video):
                os.remove(temp_video)
        except subprocess.CalledProcessError:
            if os.path.exists(temp_video):
                os.rename(temp_video, output_path)

        logger.info(f"✅ 人物替换完成: {output_path}")
        return True
