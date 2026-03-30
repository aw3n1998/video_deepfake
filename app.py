#!/usr/bin/env python3
"""
AI 视频智能重绘工作站 - Gradio 界面

两大功能：
1. 通用视频重绘 — 全局风格转换
2. 人物替换 — 多人场景中替换指定人物
"""

import argparse
import logging
import os
import sys
import threading
from pathlib import Path

try:
    import gradio as gr
except ImportError:
    print("[错误] 未安装 gradio，请运行: pip install gradio")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,0.0.0.0")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# 日志收集
# ════════════════════════════════════════════════════════════

class ProgressCollector(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []
        self._lock = threading.Lock()

    def emit(self, record):
        msg = self.format(record)
        with self._lock:
            self.lines.append(msg)
            if len(self.lines) > 300:
                self.lines = self.lines[-300:]

    def get_log(self) -> str:
        with self._lock:
            return "\n".join(self.lines)

    def clear(self):
        with self._lock:
            self.lines.clear()


_progress = ProgressCollector()
_progress.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
logging.getLogger().addHandler(_progress)

_lock = threading.Lock()
_processing = False


def get_progress():
    return _progress.get_log()


# ════════════════════════════════════════════════════════════
# Tab 1: 通用视频重绘
# ════════════════════════════════════════════════════════════

def _run_v2v(video, ref, prompt, neg, strength, steps, res, cfg, cn_scale, temporal):
    global _processing
    with _lock:
        if _processing:
            return None, "⚠️ 已有任务运行中"
        _processing = True

    if not video:
        with _lock:
            _processing = False
        return None, "❌ 请上传视频"
    if not prompt or not prompt.strip():
        with _lock:
            _processing = False
        return None, "❌ 请输入提示词"

    _progress.clear()
    out = "output_v2v.mp4"
    try:
        from src.vid2vid import Vid2VidPipeline
        from src.utils import sanitize_prompt, clamp

        pipe = Vid2VidPipeline(device="cuda")
        ok = pipe.process_video(
            video_path=video,
            output_path=out,
            prompt=sanitize_prompt(prompt),
            negative_prompt=sanitize_prompt(neg) if neg else "",
            reference_image=ref,
            num_steps=int(clamp(steps, 10, 50)),
            strength=clamp(strength, 0.1, 0.9),
            guidance_scale=clamp(cfg, 1.0, 20.0),
            controlnet_conditioning_scale=clamp(cn_scale, 0.0, 2.0),
            resolution=int(clamp(res, 512, 1024)),
            temporal_smoothing=clamp(temporal, 0.0, 0.5),
        )
        return (out if ok else None), ("✅ 重绘完成！" if ok else "❌ 失败")
    except Exception as e:
        logger.exception(f"异常: {e}")
        return None, f"❌ {e}"
    finally:
        with _lock:
            _processing = False


# ════════════════════════════════════════════════════════════
# Tab 2: 人物替换
# ════════════════════════════════════════════════════════════

_swap_pipeline = None


def _detect_persons(video_path):
    """检测视频首帧中的人物，返回人脸裁剪图列表"""
    if not video_path:
        return [], "❌ 请先上传视频"
    try:
        global _swap_pipeline
        from src.person_swap import PersonSwapPipeline
        if _swap_pipeline is None:
            _swap_pipeline = PersonSwapPipeline(device="cuda")

        crops = _swap_pipeline.detect_persons_from_video(video_path)
        if not crops:
            return [], "❌ 未检测到人物"

        # 返回带编号标注的图片
        labeled = []
        for i, crop in enumerate(crops):
            import cv2
            import numpy as np
            img = crop.copy()
            cv2.putText(img, f"#{i}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            labeled.append(img)

        return labeled, f"✅ 检测到 {len(crops)} 个人物，请选择要替换的编号"
    except Exception as e:
        logger.exception(f"检测失败: {e}")
        return [], f"❌ 检测失败: {e}"


def _run_swap(video, ref_img, person_idx, prompt, neg, strength, steps, res):
    global _processing
    with _lock:
        if _processing:
            return None, "⚠️ 已有任务运行中"
        _processing = True

    if not video:
        with _lock:
            _processing = False
        return None, "❌ 请上传视频"
    if not ref_img:
        with _lock:
            _processing = False
        return None, "❌ 请上传参考人物图片"

    _progress.clear()
    out = "output_swap.mp4"
    try:
        global _swap_pipeline
        from src.person_swap import PersonSwapPipeline
        from src.utils import sanitize_prompt, clamp

        if _swap_pipeline is None:
            _swap_pipeline = PersonSwapPipeline(device="cuda")

        ok = _swap_pipeline.process_video(
            video_path=video,
            output_path=out,
            reference_image_path=ref_img,
            target_person_index=int(person_idx),
            prompt=sanitize_prompt(prompt) if prompt else (
                "same person as reference, natural appearance, "
                "consistent lighting, high quality, realistic"
            ),
            negative_prompt=sanitize_prompt(neg) if neg else (
                "low quality, blurry, deformed, ugly, bad anatomy"
            ),
            strength=clamp(strength, 0.3, 0.9),
            num_steps=int(clamp(steps, 10, 50)),
            resolution=int(clamp(res, 512, 1024)),
        )
        return (out if ok else None), ("✅ 人物替换完成！" if ok else "❌ 失败")
    except Exception as e:
        logger.exception(f"异常: {e}")
        return None, f"❌ {e}"
    finally:
        with _lock:
            _processing = False


# ════════════════════════════════════════════════════════════
# 预设
# ════════════════════════════════════════════════════════════

PRESETS = {
    "无预设": ("", ""),
    "真实感增强": (
        "raw photo, ultra realistic, natural lighting, high detail skin, sharp focus, film grain",
        "doll-like, overly smooth, plastic, 3d render, cartoon",
    ),
    "电影质感": (
        "cinematic lighting, color grading, depth of field, film look, anamorphic lens",
        "flat lighting, amateur, oversaturated",
    ),
    "赛博朋克": (
        "cyberpunk aesthetic, neon lights, futuristic, glowing accents, rain-soaked streets",
        "daylight, natural, plain background",
    ),
    "油画风格": (
        "oil painting style, impressionist, visible brushstrokes, rich warm colors",
        "photorealistic, sharp, digital",
    ),
    "日系动漫": (
        "anime style, cel shading, vibrant colors, detailed eyes, studio ghibli quality",
        "photorealistic, western, 3d render",
    ),
}


# ════════════════════════════════════════════════════════════
# 构建 UI
# ════════════════════════════════════════════════════════════

def build_ui():
    with gr.Blocks(title="AI 视频智能重绘", theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
        # 🎬 AI 视频智能重绘工作站
        **两大模式：全局风格重绘 · 指定人物替换**
        """)

        with gr.Tabs():
            # ──────────────────────────────────────────
            # Tab 1: 通用视频重绘
            # ──────────────────────────────────────────
            with gr.Tab("🎨 通用视频重绘"):
                gr.Markdown("上传视频 + 提示词 → 全画面风格重绘（保留原始结构和动作）")

                with gr.Row():
                    with gr.Column(scale=2):
                        v2v_video = gr.Video(label="📹 输入视频", height=280)
                    with gr.Column(scale=2):
                        v2v_ref = gr.Image(label="🖼️ 参考图 (可选)", type="filepath", height=280)

                with gr.Accordion("⚙️ 提示词与参数", open=True):
                    with gr.Row():
                        v2v_prompt = gr.Textbox(
                            label="正向提示词",
                            placeholder="例如: cinematic lighting, detailed skin, warm color grading",
                            lines=3, scale=4,
                        )
                        v2v_preset = gr.Dropdown(
                            choices=list(PRESETS.keys()), label="✨ 预设", value="无预设", scale=1,
                        )
                    v2v_neg = gr.Textbox(
                        label="负向提示词",
                        value="low quality, blurry, deformed, ugly, bad anatomy, watermark",
                        lines=2,
                    )
                    with gr.Row():
                        v2v_strength = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="🎨 重绘强度")
                        v2v_steps = gr.Slider(10, 50, value=25, step=1, label="🔢 步数")
                        v2v_res = gr.Slider(512, 1024, value=768, step=64, label="📐 分辨率")
                    with gr.Row():
                        v2v_cfg = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="📏 CFG")
                        v2v_cn = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="🔒 结构锁定")
                        v2v_temporal = gr.Slider(0.0, 0.5, value=0.12, step=0.02, label="🎞️ 时序平滑")

                with gr.Row():
                    v2v_btn = gr.Button("🚀 开始重绘", variant="primary", scale=3, size="lg")
                    v2v_status = gr.Textbox(label="状态", scale=2, interactive=False)

                v2v_output = gr.Video(label="📤 输出视频", height=350)

                # 预设回调
                def on_preset(choice, cur_p, cur_n):
                    if choice in PRESETS and choice != "无预设":
                        p, n = PRESETS[choice]
                        return (f"{cur_p}, {p}" if cur_p.strip() else p,
                                f"{cur_n}, {n}" if cur_n.strip() else n)
                    return cur_p, cur_n

                v2v_preset.change(fn=on_preset, inputs=[v2v_preset, v2v_prompt, v2v_neg],
                                  outputs=[v2v_prompt, v2v_neg])

                v2v_btn.click(
                    fn=_run_v2v,
                    inputs=[v2v_video, v2v_ref, v2v_prompt, v2v_neg,
                            v2v_strength, v2v_steps, v2v_res, v2v_cfg, v2v_cn, v2v_temporal],
                    outputs=[v2v_output, v2v_status],
                )

            # ──────────────────────────────────────────
            # Tab 2: 人物替换
            # ──────────────────────────────────────────
            with gr.Tab("👤 人物替换"):
                gr.Markdown(
                    "**多人场景指定替换**: 上传视频 → 检测人物 → 选择目标 → "
                    "上传参考人物图 → 仅替换该人物，其他人和背景不变"
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        swap_video = gr.Video(label="📹 输入视频", height=280)
                    with gr.Column(scale=2):
                        swap_ref = gr.Image(label="🖼️ 目标人物参考图", type="filepath", height=280)

                with gr.Row():
                    swap_detect_btn = gr.Button("🔍 检测人物", variant="secondary", scale=1)
                    swap_detect_status = gr.Textbox(label="检测结果", scale=2, interactive=False)

                swap_gallery = gr.Gallery(
                    label="检测到的人物（带编号）",
                    columns=6, rows=1, height=150,
                    object_fit="contain",
                )

                with gr.Accordion("⚙️ 替换参数", open=True):
                    with gr.Row():
                        swap_idx = gr.Number(
                            label="替换目标编号 (#)", value=0, precision=0,
                            minimum=0, maximum=10,
                        )
                        swap_strength = gr.Slider(
                            0.3, 0.9, value=0.65, step=0.05,
                            label="🎨 替换强度 (推荐0.55~0.70)",
                        )
                        swap_steps = gr.Slider(10, 50, value=25, step=1, label="🔢 步数")
                        swap_res = gr.Slider(512, 1024, value=768, step=64, label="📐 分辨率")

                    swap_prompt = gr.Textbox(
                        label="提示词 (可选，留空使用默认)",
                        placeholder="留空即可，或输入: realistic, natural skin, consistent lighting",
                        lines=2,
                    )
                    swap_neg = gr.Textbox(
                        label="负向提示词 (可选)",
                        value="low quality, blurry, deformed, ugly",
                        lines=1,
                    )

                with gr.Row():
                    swap_btn = gr.Button("🚀 开始替换", variant="primary", scale=3, size="lg")
                    swap_status = gr.Textbox(label="状态", scale=2, interactive=False)

                swap_output = gr.Video(label="📤 替换结果", height=350)

                # 检测回调
                swap_detect_btn.click(
                    fn=_detect_persons,
                    inputs=[swap_video],
                    outputs=[swap_gallery, swap_detect_status],
                )

                # 替换回调
                swap_btn.click(
                    fn=_run_swap,
                    inputs=[swap_video, swap_ref, swap_idx,
                            swap_prompt, swap_neg, swap_strength, swap_steps, swap_res],
                    outputs=[swap_output, swap_status],
                )

        # ──────────────────────────────────────────
        # 日志
        # ──────────────────────────────────────────
        with gr.Accordion("📋 实时日志", open=False):
            log_box = gr.Textbox(label="", lines=15, interactive=False, autoscroll=True)
            gr.Timer(2.0).tick(fn=get_progress, outputs=log_box)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI 视频智能重绘工作站")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--auth", nargs=2, metavar=("USER", "PASS"), help="登录认证")
    args = parser.parse_args()

    demo = build_ui()
    kwargs = dict(server_name=args.host, server_port=args.port, share=args.share)
    if args.auth:
        kwargs["auth"] = tuple(args.auth)
    demo.launch(**kwargs)
