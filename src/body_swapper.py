"""
全身重绘模块 - 使用 ControlNet + Stable Diffusion 将男性人物重绘为女性
需要 GPU (推荐 8GB+ 显存，Colab T4 可用)

流程:
  1. InsightFace 检测帧中是否有男性人脸
  2. MediaPipe 提取人体分割掩码
  3. controlnet_aux OpenposeDetector 提取姿态骨架
  4. ControlNet + SD img2img 重绘为女性外观
  5. 将重绘结果与原始背景按掩码融合
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional
import torch

logger = logging.getLogger(__name__)


class BodySwapper:
    """
    ControlNet + SD 全身性别重绘器

    参数:
        sd_model      : SD 模型 HuggingFace ID
        controlnet_model : ControlNet 模型 ID
        device        : 'cuda' 或 'cpu'
        strength      : img2img 重绘强度 (0.6~0.85)，越高变化越大
        guidance_scale: CFG 引导强度
    """

    # 正面提示词
    DEFAULT_PROMPT = (
        "beautiful young woman, female, long hair, feminine appearance, "
        "photorealistic, high quality, 8k uhd, professional photography, "
        "natural lighting, detailed face"
    )
    # 负面提示词
    DEFAULT_NEG = (
        "male, man, boy, masculine, beard, mustache, stubble, "
        "ugly, deformed, blurry, low quality, watermark, nsfw, "
        "extra limbs, bad anatomy"
    )

    def __init__(
        self,
        sd_model: str = "SG161222/Realistic_Vision_V5.1_noVAE",
        controlnet_model: str = "lllyasviel/control_v11p_sd15_openpose",
        device: str = "cuda",
        strength: float = 0.75,
        guidance_scale: float = 7.5,
    ):
        self.device = device
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.sd_model = sd_model
        self.controlnet_model = controlnet_model

        self.pipe = None
        self.pose_detector = None
        self.seg_model = None
        self.face_analyser = None

        self._init_models()

    # ------------------------------------------------------------------
    # 模型初始化
    # ------------------------------------------------------------------

    def _init_models(self):
        self._init_face_analyser()
        self._init_pose_detector()
        self._init_segmentation()
        self._init_sd_pipeline()

    def _init_face_analyser(self):
        """InsightFace 性别检测"""
        import insightface
        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if self.device == 'cuda'
            else ['CPUExecutionProvider']
        )
        self.face_analyser = insightface.app.FaceAnalysis(
            name='buffalo_l', providers=providers
        )
        self.face_analyser.prepare(ctx_id=0 if self.device == 'cuda' else -1,
                                   det_size=(640, 640))
        logger.info("[OK] InsightFace 加载成功")

    def _init_pose_detector(self):
        """OpenPose 姿态检测器"""
        from controlnet_aux import OpenposeDetector
        self.pose_detector = OpenposeDetector.from_pretrained(
            'lllyasviel/ControlNet'
        )
        logger.info("[OK] OpenPose 检测器加载成功")

    def _init_segmentation(self):
        """MediaPipe 人体分割"""
        try:
            import mediapipe as mp
            self._mp_seg = mp.solutions.selfie_segmentation
            self.seg_model = self._mp_seg.SelfieSegmentation(model_selection=1)
            logger.info("[OK] MediaPipe 人体分割加载成功")
        except Exception as e:
            logger.warning(f"MediaPipe 分割不可用，将使用全帧掩码: {e}")
            self.seg_model = None

    def _init_sd_pipeline(self):
        """初始化 ControlNet img2img pipeline"""
        from diffusers import (
            StableDiffusionControlNetImg2ImgPipeline,
            ControlNetModel,
            UniPCMultistepScheduler,
        )

        logger.info(f"加载 ControlNet: {self.controlnet_model} ...")
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model,
            torch_dtype=torch.float16,
        )

        logger.info(f"加载 SD 模型: {self.sd_model} ...")
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.sd_model,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(self.device)

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        # 显存优化
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            logger.info("[OK] xformers 显存优化已启用")
        except Exception:
            pass

        logger.info("[OK] SD + ControlNet img2img pipeline 加载成功")

    # ------------------------------------------------------------------
    # 单帧处理
    # ------------------------------------------------------------------

    def has_male(self, frame: np.ndarray) -> bool:
        """检测帧中是否有男性人脸 (gender==1)"""
        try:
            faces = self.face_analyser.get(frame)
            return any(getattr(f, 'gender', 0) == 1 for f in faces)
        except Exception:
            return False

    def extract_pose(self, frame: np.ndarray) -> np.ndarray:
        """提取人体姿态骨架图（RGB numpy）"""
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pose_pil = self.pose_detector(pil)
        return np.array(pose_pil)

    def segment_person(self, frame: np.ndarray) -> np.ndarray:
        """
        返回人体二值掩码 (uint8, 0/255)
        白色=人体，黑色=背景
        """
        if self.seg_model is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.seg_model.process(rgb)
            mask = (result.segmentation_mask > 0.5).astype(np.uint8) * 255
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            return mask
        # 备用：全帧
        return np.full(frame.shape[:2], 255, dtype=np.uint8)

    def generate_female(
        self,
        frame: np.ndarray,
        pose_image: np.ndarray,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
    ) -> np.ndarray:
        """
        用 ControlNet img2img 生成女性版本
        返回 BGR numpy 数组，与输入同尺寸
        """
        from PIL import Image

        prompt = prompt or self.DEFAULT_PROMPT
        negative_prompt = negative_prompt or self.DEFAULT_NEG

        h, w = frame.shape[:2]
        # SD 要求宽高是 8 的倍数
        gen_w = max((w // 8) * 8, 512)
        gen_h = max((h // 8) * 8, 512)

        init_img = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ).resize((gen_w, gen_h))

        ctrl_img = Image.fromarray(pose_image).resize((gen_w, gen_h))

        with torch.autocast(self.device):
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_img,
                control_image=ctrl_img,
                strength=self.strength,
                num_inference_steps=25,
                guidance_scale=self.guidance_scale,
                controlnet_conditioning_scale=0.8,
            ).images[0]

        result_np = np.array(result.resize((w, h)))
        return cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

    def blend(
        self,
        original: np.ndarray,
        generated: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """将生成的女性人物与原始背景融合（羽化边缘）"""
        m = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 0) / 255.0
        m3 = m[:, :, np.newaxis]
        blended = generated.astype(np.float32) * m3 + original.astype(np.float32) * (1 - m3)
        return blended.astype(np.uint8)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理单帧：检测 → 重绘 → 融合"""
        if not self.has_male(frame):
            return frame  # 无男性，直接返回原帧

        pose = self.extract_pose(frame)
        mask = self.segment_person(frame)
        generated = self.generate_female(frame, pose)
        return self.blend(frame, generated, mask)

    # ------------------------------------------------------------------
    # 视频处理
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        output_path: str,
        skip_frames: int = 2,
    ) -> bool:
        """
        处理整个视频

        Args:
            video_path  : 输入视频路径
            output_path : 输出视频路径（无音频）
            skip_frames : 跳帧数，提速用
                          0 = 每帧都重绘（最慢，最好）
                          2 = 每3帧重绘1帧（推荐，速度快3倍）
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {video_path}")
            return False

        fps   = cap.get(cv2.CAP_PROP_FPS)
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_idx = 0
        last_processed = None

        logger.info(f"视频信息: {total} 帧 | {fps:.1f}fps | {w}x{h}")
        logger.info(f"跳帧模式: skip_frames={skip_frames} "
                    f"(实际处理约 {total//(skip_frames+1)} 帧)")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    # 复用上一帧结果，速度快但运动帧有拖影
                    out.write(last_processed if last_processed is not None else frame)
                else:
                    processed = self.process_frame(frame)
                    last_processed = processed
                    out.write(processed)

                frame_idx += 1
                if frame_idx % 30 == 0:
                    pct = frame_idx / total * 100
                    logger.info(f"进度: {frame_idx}/{total} ({pct:.1f}%)")

        finally:
            cap.release()
            out.release()

        logger.info(f"[OK] 视频重绘完成: {output_path}")
        return True
