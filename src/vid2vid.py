"""
高质量视频转视频模块 (High-Fidelity Vid2Vid)

核心修复：
- 旧版使用 AnimateDiffControlNetPipeline (text-to-video)，完全忽略原始帧内容
- 新版使用 StableDiffusionControlNetImg2ImgPipeline (image-to-image)，
  将原始帧作为输入，通过 strength 控制保留程度

技术栈：
- Stable Diffusion 1.5 img2img + Depth ControlNet
- IP-Adapter (可选，用于参考图引导)
- 颜色校正 (LAB 色彩空间迁移)
- 时序平滑 (帧间混合 + 亮度去闪烁)
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
from typing import List, Optional

logger = logging.getLogger(__name__)

# 时序一致性参数
DEFLICKER_WINDOW = 7
DEFAULT_STRENGTH = 0.35  # 低强度 = 保留更多原始内容 = 更自然


class Vid2VidPipeline:
    """
    正确的 Video-to-Video 管线。

    与旧版的关键区别：
    - 旧版: AnimateDiff (text-to-video) → 生成与原视频完全无关的内容
    - 新版: img2img + ControlNet → 以原始帧为基础，按提示词微调

    strength 参数说明：
    - 0.2 ~ 0.3: 微调 (色彩/光照调整，几乎不变)
    - 0.3 ~ 0.5: 风格迁移 (保留结构，改变风格) ← 推荐
    - 0.5 ~ 0.7: 中度重绘 (明显改变，保留大致轮廓)
    - 0.7 ~ 1.0: 大幅重绘 (几乎从零生成) ← 旧版默认值，导致AI感严重
    """

    def __init__(
        self,
        sd_model: str = "SG161222/Realistic_Vision_V5.1_noVAE",
        device: str = "cuda",
    ):
        self.device = device
        self.sd_model = sd_model
        self.pipe = None
        self.depth_estimator = None
        self._ip_adapter_loaded = False

    def _init_models(self):
        """初始化 img2img + ControlNet 管线"""
        from diffusers import (
            StableDiffusionControlNetImg2ImgPipeline,
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

        logger.info(f"加载 Stable Diffusion: {self.sd_model}")
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.sd_model,
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
            logger.info("xformers 显存优化已启用")
        except Exception:
            logger.info("xformers 不可用，使用默认注意力机制")

        logger.info("✅ 管线初始化完成")

    def _load_ip_adapter(self):
        """加载 IP-Adapter 用于参考图引导"""
        if self._ip_adapter_loaded:
            return True
        try:
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter_sd15.bin",
            )
            self.pipe.set_ip_adapter_scale(0.4)
            self._ip_adapter_loaded = True
            logger.info("✅ IP-Adapter 已加载（参考图引导可用）")
            return True
        except Exception as e:
            logger.warning(f"IP-Adapter 加载失败，将忽略参考图: {e}")
            return False

    def _estimate_depth(self, pil_img: Image.Image) -> Image.Image:
        """提取深度图用于 ControlNet 结构引导"""
        result = self.depth_estimator(pil_img)
        return result["depth"].convert("RGB")

    @staticmethod
    def _color_transfer(generated: np.ndarray, original: np.ndarray) -> np.ndarray:
        """
        颜色校正：将原始帧的颜色统计量迁移到生成帧。
        使用 LAB 色彩空间，分别对 L/A/B 三通道做均值-标准差匹配。
        """
        src_lab = cv2.cvtColor(generated, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)

        for i in range(3):
            src_mean, src_std = src_lab[:, :, i].mean(), src_lab[:, :, i].std()
            tgt_mean, tgt_std = tgt_lab[:, :, i].mean(), tgt_lab[:, :, i].std()
            if src_std > 1e-6:
                src_lab[:, :, i] = (
                    (src_lab[:, :, i] - src_mean) * (tgt_std / src_std) + tgt_mean
                )

        return cv2.cvtColor(
            np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
        )

    @staticmethod
    def _temporal_blend(
        prev_frame: np.ndarray, curr_frame: np.ndarray, alpha: float = 0.12
    ) -> np.ndarray:
        """帧间时序混合：减少帧间突变"""
        return cv2.addWeighted(curr_frame, 1.0 - alpha, prev_frame, alpha, 0)

    @staticmethod
    def _deflicker(frames: List[np.ndarray], window: int = DEFLICKER_WINDOW) -> List[np.ndarray]:
        """
        亮度去闪烁：平滑帧间亮度曲线。
        对每帧的平均亮度做滑动窗口平滑，然后按比例缩放。
        """
        if len(frames) < 3:
            return frames

        luminances = []
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            luminances.append(float(gray.mean()))

        # 滑动窗口平滑
        smoothed = []
        for i in range(len(luminances)):
            start = max(0, i - window // 2)
            end = min(len(luminances), i + window // 2 + 1)
            smoothed.append(np.mean(luminances[start:end]))

        result = []
        for i, f in enumerate(frames):
            if luminances[i] > 1e-6:
                factor = smoothed[i] / luminances[i]
                # 限制缩放范围，避免异常
                factor = np.clip(factor, 0.8, 1.2)
                corrected = np.clip(
                    f.astype(np.float32) * factor, 0, 255
                ).astype(np.uint8)
                result.append(corrected)
            else:
                result.append(f)

        return result

    def process_video(
        self,
        video_path: str,
        output_path: str,
        prompt: str,
        negative_prompt: str = (
            "low quality, blurry, deformed, ugly, bad anatomy, "
            "disfigured, watermark, text, signature, jpeg artifacts"
        ),
        reference_image: Optional[str] = None,
        num_steps: int = 25,
        strength: float = DEFAULT_STRENGTH,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.8,
        resolution: int = 768,
        color_correction: bool = True,
        temporal_smoothing: float = 0.12,
    ) -> bool:
        """
        核心处理函数：逐帧 img2img + ControlNet。

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            prompt: 正向提示词（描述想要的效果）
            negative_prompt: 负向提示词
            reference_image: 参考图路径（可选，通过 IP-Adapter 注入视觉风格）
            num_steps: 推理步数（越高质量越好，速度越慢）
            strength: 重绘强度（0.2~0.5 推荐，越低越接近原视频）
            guidance_scale: CFG 引导系数
            controlnet_conditioning_scale: ControlNet 结构锁定强度
            resolution: 处理分辨率
            color_correction: 是否启用颜色校正
            temporal_smoothing: 时序平滑强度（0=关闭）

        Returns:
            成功返回 True
        """
        # 延迟初始化模型
        if self.pipe is None:
            self._init_models()

        # 输入验证
        if not os.path.isfile(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            return False

        # 处理参考图（IP-Adapter）
        ip_adapter_image = None
        if reference_image and os.path.isfile(reference_image):
            logger.info(f"检测到参考图: {reference_image}")
            if self._load_ip_adapter():
                ip_adapter_image = Image.open(reference_image).convert("RGB")
                logger.info("参考图已加载到 IP-Adapter，将引导生成风格")
            else:
                logger.info("IP-Adapter 不可用，参考图将被忽略")

        final_prompt = prompt

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
        logger.info(f"视频信息: {total} 帧 | {fps:.1f}fps | {orig_w}x{orig_h}")

        # 计算处理分辨率（8的倍数）
        scale = resolution / max(orig_w, orig_h)
        proc_w = max(int(orig_w * scale) // 8 * 8, 512)
        proc_h = max(int(orig_h * scale) // 8 * 8, 512)
        logger.info(f"处理分辨率: {proc_w}x{proc_h} | 强度: {strength}")

        # 逐帧处理
        generated_frames = []
        prev_generated = None
        fixed_generator = torch.Generator(device=self.device).manual_seed(42)

        for i, frame in enumerate(frames):
            # 转 PIL 并缩放
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_resized = pil_frame.resize((proc_w, proc_h), Image.LANCZOS)

            # 提取深度图
            depth_map = self._estimate_depth(pil_resized)

            # 构建管线参数
            pipe_kwargs = dict(
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                image=pil_resized,  # ✅ 关键：原始帧作为输入
                control_image=depth_map,
                num_inference_steps=num_steps,
                strength=strength,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=fixed_generator,
            )

            # IP-Adapter 参考图注入
            if ip_adapter_image is not None:
                pipe_kwargs["ip_adapter_image"] = ip_adapter_image

            # 推理
            with torch.autocast(self.device):
                result = self.pipe(**pipe_kwargs)

            # 后处理：缩放回原分辨率
            gen_pil = result.images[0]
            gen_frame = cv2.cvtColor(
                np.array(gen_pil.resize((orig_w, orig_h), Image.LANCZOS)),
                cv2.COLOR_RGB2BGR,
            )

            # 颜色校正：将生成帧的颜色匹配到原始帧
            if color_correction:
                gen_frame = self._color_transfer(gen_frame, frame)

            # 时序平滑：与上一帧混合
            if prev_generated is not None and temporal_smoothing > 0:
                gen_frame = self._temporal_blend(
                    prev_generated, gen_frame, temporal_smoothing
                )

            generated_frames.append(gen_frame)
            prev_generated = gen_frame

            # 进度日志
            if (i + 1) % 5 == 0 or i == 0 or i == total - 1:
                pct = 100 * (i + 1) / total
                logger.info(f"  帧 {i+1}/{total} ({pct:.0f}%)")

            # 定期释放显存
            if (i + 1) % 30 == 0:
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        # 亮度去闪烁
        logger.info("应用亮度去闪烁...")
        generated_frames = self._deflicker(generated_frames)

        # 写入临时视频文件
        temp_video = output_path.replace(".mp4", "_temp_noaudio.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_video, fourcc, fps, (orig_w, orig_h))
        for f in generated_frames:
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
            logger.info(f"✅ 视频处理完成: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"FFmpeg 音频合并失败: {e}")
            if os.path.exists(temp_video):
                os.rename(temp_video, output_path)
                logger.info(f"已保存无音频版本: {output_path}")

        return True
