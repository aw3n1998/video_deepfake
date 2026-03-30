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

# 组合关键词：prompt 同时包含任意一个"动作词"和任意一个"头发词"即触发
_HAIR_WORDS = ["头发", "发丝", "hair", "strand"]
_FALL_WORDS = ["掉", "落", "脱", "飘", "fall", "drop", "los"]


def should_trigger(prompt: str) -> bool:
    """检测提示词是否包含掉发关键词，决定是否自动触发。

    支持两种匹配：
    1. 精确子串匹配 (如 "掉发", "hair fall")
    2. 组合匹配: 同时包含头发词 + 掉落词 (如 "掉落了一些头发")
    """
    if not prompt:
        return False
    prompt_lower = prompt.lower()
    # 精确匹配
    if any(kw in prompt_lower for kw in HAIR_FALL_KEYWORDS):
        return True
    # 组合匹配：同时含"头发相关词"和"掉落相关词"
    has_hair = any(w in prompt_lower for w in _HAIR_WORDS)
    has_fall = any(w in prompt_lower for w in _FALL_WORDS)
    return has_hair and has_fall


# 视觉风格关键词：如果提示词含有这些，说明用户同时想改变画面风格
_STYLE_WORDS = [
    "风格", "质感", "光照", "色调", "赛博", "朋克", "油画", "动漫", "写实",
    "电影", "复古", "梦幻", "水彩", "卡通", "黑白", "暖色", "冷色",
    "style", "cinematic", "anime", "realistic", "painting", "vintage",
    "cyberpunk", "fantasy", "watercolor", "cartoon",
]


def is_hair_only(prompt: str) -> bool:
    """判断提示词是否纯粹描述掉发，不含视觉风格指令。

    纯掉发描述（如 "梳头发的时候慢慢掉落了一些头发"）不需要经过 vid2vid，
    直接叠加粒子特效即可，避免画质劣化。
    """
    if not prompt:
        return True
    prompt_lower = prompt.lower()
    return not any(w in prompt_lower for w in _STYLE_WORDS)


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
    angular_velocity: float
    color: Tuple[int, int, int]
    opacity: float
    lifetime: int
    age: int = 0


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
            # 显式导入以防属性丢失 (Colab 的 protobuf 冲突常导致 mp.solutions 丢失)
            import mediapipe.python.solutions.face_detection as mp_face_detection
            
            self._mp_detector = mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            logger.info("掉发特效: 使用 MediaPipe 检测")
        except Exception as e:
            logger.warning(f"MediaPipe 无法初始化 ({e})，使用 OpenCV Haar 退化方案")

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

        # 降级方案：寻找画面中下部的深色大色块（大概率是头发）
        # 因拍摄角度常为俯视或后脑勺，不露脸时有效避免毛发飘在半空
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 只在画面中下部寻找，避免取到边缘或顶部背景
        mask = np.zeros_like(gray)
        cv2.ellipse(mask, (w//2, h//2 + int(h*0.05)), (int(w*0.35), int(h*0.35)), 0, 0, 360, 255, -1)
        gray_masked = cv2.bitwise_and(gray, mask)
        
        # 寻找比较暗的区域 (假设头发阈值 < 80)
        _, thresh = cv2.threshold(gray_masked, 80, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.bitwise_and(thresh, mask)  # 再次排除掩码外区域
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > (w * h * 0.02):  # 面积足够大
                bx, by, bw, bh = cv2.boundingRect(c)
                # 返回深色区域的上边缘略靠下作为掉发中心
                return (bx + bw // 2, by + int(bh * 0.15), bw, bh)

        # 最终降级：画面中间偏下 (不再默认 h//5 的天花板位置)
        return (w // 2, int(h * 0.4), w // 3, h // 4)

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
        cv = 10
        c = tuple(max(0, min(255, ch + random.randint(-cv, cv))) for ch in color)

        return HairStrand(
            x=cx + random.gauss(0, fw * 0.35),
            y=ty + random.uniform(-fh * 0.1, fh * 0.25),
            vx=random.gauss(0, 0.8),
            vy=random.uniform(0.1, 1.0),
            length=random.uniform(8, 25) * length_mult,  # 缩短长度，变小碎发
            thickness=random.uniform(0.5, 1.2),          # 变细
            curl=random.uniform(-0.8, 0.8),              # 随机两端卷曲方向
            angle=random.uniform(0, math.pi * 2),        # 起始翻滚角
            angular_velocity=random.gauss(0, 0.15),      # 翻滚速度
            color=c,
            opacity=random.uniform(0.6, 0.85),           # 增加透明感
            lifetime=int(random.uniform(50, 150)),
        )

    @staticmethod
    def _update(s: HairStrand, speed: float) -> bool:
        """更新物理状态，返回 False 表示已死"""
        s.vy += 0.25 * speed  # 重力增强
        s.vx *= 0.95          # 空气阻力
        s.vy *= 0.96          # 空气阻力
        s.vx += random.gauss(0, 0.3)  # 风吹扰动
        s.x += s.vx * speed
        s.y += s.vy * speed
        s.angle += s.angular_velocity * speed  # 空中翻滚
        
        s.age += 1
        s.lifetime -= 1
        if s.lifetime < 20:
            s.opacity *= 0.85
        return s.lifetime > 0 and s.opacity > 0.01

    @staticmethod
    def _draw(frame: np.ndarray, s: HairStrand):
        """渲染一根细软飞扬的发丝 (弧线)"""
        x0, y0 = s.x, s.y
        # 从中心点计算两端的偏移量
        dx = math.cos(s.angle) * s.length / 2
        dy = math.sin(s.angle) * s.length / 2
        
        p1_x, p1_y = int(x0 - dx), int(y0 - dy)
        p2_x, p2_y = int(x0 + dx), int(y0 + dy)
        
        # 贝塞尔控制点: 向法线方向偏移制造出卷曲弧度
        nx = -math.sin(s.angle) * s.length * s.curl
        ny = math.cos(s.angle) * s.length * s.curl
        ctrl_x = int(x0 + nx)
        ctrl_y = int(y0 + ny)

        pts = []
        n = 6  # 分段数，因为发丝短，不需要太极度平滑
        for ti in range(n + 1):
            t = ti / n
            bx = (1 - t) ** 2 * p1_x + 2 * (1 - t) * t * ctrl_x + t ** 2 * p2_x
            by = (1 - t) ** 2 * p1_y + 2 * (1 - t) * t * ctrl_y + t ** 2 * p2_y
            pts.append((int(bx), int(by)))

        if len(pts) < 2:
            return

        overlay = frame.copy()
        t_thick = 1 if s.thickness <= 1.5 else 2
        
        for j in range(len(pts) - 1):
            cv2.line(overlay, pts[j], pts[j + 1], s.color, t_thick, cv2.LINE_AA)

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
