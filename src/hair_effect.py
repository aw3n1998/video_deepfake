"""
掉发特效模块 (Hair Fall Effect)

纯粹的物理粒子系统，不依赖 AI，完全解耦。
自动检测头部位置 → 生成弯曲发丝粒子 → 物理模拟重力掉落 → 合成到原视频。

触发方式:
  1. 独立标签页手动使用
  2. 自动检测: 提示词包含 "掉发/掉头发/hair fall" 等关键词时,
     vid2vid 完成后自动叠加此特效作为后处理

不需要显卡，纯 CPU 运算，速度极快。
"""

import cv2
import math
import logging
import os
import random
import subprocess
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════
# 关键词检测（解耦触发）
# ════════════════════════════════════════════════════════════

HAIR_FALL_KEYWORDS = [
    "掉发", "掉头发", "脱发", "头发掉", "头发脱落", "头发从头上掉",
    "梳头掉", "洗头掉", "hair fall", "hair loss", "hair falling",
    "losing hair", "strands falling", "hair strands falling",
]


def should_trigger(prompt: str) -> bool:
    """检测提示词是否包含掉发关键词，决定是否自动触发"""
    if not prompt:
        return False
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in HAIR_FALL_KEYWORDS)


# ════════════════════════════════════════════════════════════
# 发丝粒子
# ════════════════════════════════════════════════════════════

@dataclass
class HairStrand:
    """一根头发丝的物理状态"""
    x: float
    y: float
    vx: float
    vy: float
    length: float
    thickness: float
    curl: float
    angle: float
    color: Tuple[int, int, int]
    opacity: float
    lifetime: int
    age: int = 0
    ctrl_ox: float = 0.0
    ctrl_oy: float = 0.0


# ════════════════════════════════════════════════════════════
# 掉发特效引擎
# ════════════════════════════════════════════════════════════

class HairFallEffect:
    """
    掉发粒子特效。

    原理: 检测头部位置 → 在头顶生成发丝粒子 →
    模拟重力/空气阻力掉落 → 贝塞尔曲线渲染 → 合成到原帧。
    """

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._mp_detector = None
        self._try_mediapipe()

    def _try_mediapipe(self):
        """尝试加载 MediaPipe（更准确），失败时回退到 Haar"""
        try:
            import mediapipe as mp
            self._mp_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            logger.info("掉发特效: 使用 MediaPipe 检测")
        except ImportError:
            logger.info("掉发特效: 使用 OpenCV Haar 检测 (pip install mediapipe 可提升精度)")

    # ────────────────────────────────────────────────────────
    # 头部检测
    # ────────────────────────────────────────────────────────

    def _detect_head(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """检测头部区域 → (中心x, 顶部y, 宽, 高)"""
        h, w = frame.shape[:2]

        if self._mp_detector is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self._mp_detector.process(rgb)
            if res.detections:
                bb = res.detections[0].location_data.relative_bounding_box
                fx, fy = int(bb.xmin * w), int(bb.ymin * h)
                fw, fh = int(bb.width * w), int(bb.height * h)
                return (fx + fw // 2, max(0, fy - fh // 3), fw, fh)

        # Haar fallback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            fx, fy, fw, fh = faces[0]
            return (fx + fw // 2, max(0, fy - fh // 3), fw, fh)

        # 未检测到 → 画面上部中央
        return (w // 2, h // 5, w // 3, h // 4)

    def _sample_hair_color(
        self, frame: np.ndarray, head: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int]:
        """从头顶区域采样头发颜色（取较暗像素的均值）"""
        cx, ty, fw, fh = head
        h, w = frame.shape[:2]

        y1 = max(0, ty)
        y2 = min(h, ty + fh // 3)
        x1 = max(0, cx - fw // 3)
        x2 = min(w, cx + fw // 3)

        if y2 > y1 and x2 > x1:
            roi = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            dark = hsv[:, :, 2] < 120
            if dark.any():
                avg = roi[dark].mean(axis=0).astype(int)
                return (int(avg[0]), int(avg[1]), int(avg[2]))

        return (25, 20, 15)  # 默认黑发

    # ────────────────────────────────────────────────────────
    # 粒子生成与物理
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _spawn(
        head: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        length_mult: float,
    ) -> HairStrand:
        cx, ty, fw, fh = head
        cv = 15
        c = tuple(max(0, min(255, ch + random.randint(-cv, cv))) for ch in color)

        return HairStrand(
            x=cx + random.gauss(0, fw * 0.4),
            y=ty + random.uniform(-fh * 0.1, fh * 0.35),
            vx=random.gauss(0, 1.2),
            vy=random.uniform(0.3, 1.8),
            length=random.uniform(30, 80) * length_mult,
            thickness=random.uniform(1.0, 2.2),
            curl=random.uniform(0.2, 0.8),
            angle=random.uniform(-0.5, 0.5),
            color=c,
            opacity=random.uniform(0.55, 0.92),
            lifetime=int(random.uniform(45, 130)),
            ctrl_ox=random.gauss(0, 12),
            ctrl_oy=random.uniform(5, 22),
        )

    @staticmethod
    def _update(s: HairStrand, speed: float) -> bool:
        """更新物理状态，返回 False 表示已死"""
        s.vy += 0.15 * speed
        s.vx *= 0.98
        s.vy *= 0.99
        s.vx += random.gauss(0, 0.25)
        s.x += s.vx * speed
        s.y += s.vy * speed
        s.angle += random.gauss(0, 0.015)
        s.ctrl_ox += random.gauss(0, 0.4)
        s.age += 1
        s.lifetime -= 1
        if s.lifetime < 15:
            s.opacity *= 0.90
        return s.lifetime > 0 and s.opacity > 0.04

    @staticmethod
    def _draw(frame: np.ndarray, s: HairStrand):
        """渲染一根贝塞尔曲线发丝"""
        x0, y0 = int(s.x), int(s.y)
        dx = math.sin(s.angle) * s.length
        dy = math.cos(s.angle) * s.length
        x2, y2 = int(s.x + dx), int(s.y + dy)
        ctrl_x = int((x0 + x2) / 2 + s.ctrl_ox * s.curl)
        ctrl_y = int((y0 + y2) / 2 + s.ctrl_oy)

        n = max(8, int(s.length / 4))
        pts = []
        for ti in range(n + 1):
            t = ti / n
            bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * ctrl_x + t ** 2 * x2
            by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * ctrl_y + t ** 2 * y2
            pts.append((int(bx), int(by)))

        if len(pts) < 2:
            return

        overlay = frame.copy()
        base_t = max(1, int(s.thickness))
        for j in range(len(pts) - 1):
            progress = j / len(pts)
            t = max(1, int(base_t * (1.0 - progress * 0.5)))
            cv2.line(overlay, pts[j], pts[j + 1], s.color, t, cv2.LINE_AA)

        cv2.addWeighted(overlay, s.opacity, frame, 1.0 - s.opacity, 0, frame)

    # ────────────────────────────────────────────────────────
    # 主处理
    # ────────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        output_path: str,
        intensity: float = 1.0,
        strand_length: float = 1.0,
        spawn_rate: float = 1.0,
        fall_speed: float = 1.0,
    ) -> bool:
        """
        给视频添加掉发特效。

        Args:
            video_path: 输入视频
            output_path: 输出视频
            intensity: 掉发密度 (0.3~3.0)
            strand_length: 发丝长度倍率
            spawn_rate: 生成速率倍率
            fall_speed: 掉落速度倍率
        """
        if not os.path.isfile(video_path):
            logger.error(f"视频不存在: {video_path}")
            return False

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
        logger.info(f"掉发特效: {total} 帧 | {fps:.1f}fps | {ow}x{oh}")

        head = self._detect_head(frames[0])
        hair_color = self._sample_hair_color(frames[0], head)
        logger.info(f"  头部: {head} | 发色: {hair_color}")

        strands: List[HairStrand] = []
        base_spawn = max(1, int(2.0 * intensity * spawn_rate))
        output_frames = []

        for i, frame in enumerate(frames):
            result = frame.copy()

            # 每 5 帧更新头部位置
            if i % 5 == 0:
                new_head = self._detect_head(frame)
                if new_head:
                    head = new_head

            # 渐入 (前15帧逐渐增多)
            ramp = min(1.0, i / 15.0)
            n_spawn = int(base_spawn * ramp)
            if random.random() < 0.3 * intensity:
                n_spawn += random.randint(1, 3)

            for _ in range(n_spawn):
                strands.append(self._spawn(head, hair_color, strand_length))

            alive = []
            for s in strands:
                if self._update(s, fall_speed):
                    if -50 < s.x < ow + 50 and -50 < s.y < oh + 50:
                        self._draw(result, s)
                    alive.append(s)
            strands = alive

            output_frames.append(result)

            if (i + 1) % 30 == 0 or i == 0:
                logger.info(f"  帧 {i+1}/{total} | 活跃发丝: {len(strands)}")

        # 写入
        temp = output_path.replace(".mp4", "_temp_fx.mp4")
        writer = cv2.VideoWriter(
            temp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh)
        )
        for f in output_frames:
            writer.write(f)
        writer.release()

        # 音频合并
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg", "-i", temp, "-i", video_path,
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k",
                    "-map", "0:v:0", "-map", "1:a:0?",
                    "-shortest", "-y", output_path,
                ],
                check=True,
                capture_output=True,
            )
            if os.path.exists(temp):
                os.remove(temp)
        except subprocess.CalledProcessError:
            if os.path.exists(temp):
                os.rename(temp, output_path)

        logger.info(f"✅ 掉发特效完成: {output_path}")
        return True
