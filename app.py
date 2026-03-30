#!/usr/bin/env python3
"""
AI 视频智能重绘工作站 - Gradio 界面

功能：上传视频 + 提示词 → AI 重绘视频（保留原视频内容）
可选：参考图引导（通过 IP-Adapter 注入视觉风格）
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
# 进度日志收集
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


# ════════════════════════════════════════════════════════════
# 核心处理
# ════════════════════════════════════════════════════════════

_lock = threading.Lock()
_processing = False


def get_progress():
    return _progress.get_log()


def _run_v2v(video, ref, prompt, neg, strength, steps, res, cfg, cn_scale, temporal):
    """执行视频重绘"""
    global _processing

    # 线程安全检查
    with _lock:
        if _processing:
            return None, "⚠️ 已有任务运行中，请等待..."
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

        # 参数安全校验
        prompt = sanitize_prompt(prompt)
        neg = sanitize_prompt(neg) if neg else ""
        strength = clamp(strength, 0.1, 0.9)
        steps = int(clamp(steps, 10, 50))
        res = int(clamp(res, 512, 1024))
        cfg = clamp(cfg, 1.0, 20.0)
        cn_scale = clamp(cn_scale, 0.0, 2.0)
        temporal = clamp(temporal, 0.0, 0.5)

        pipe = Vid2VidPipeline(device="cuda")
        ok = pipe.process_video(
            video_path=video,
            output_path=out,
            prompt=prompt,
            negative_prompt=neg,
            reference_image=ref,
            num_steps=steps,
            strength=strength,
            guidance_scale=cfg,
            controlnet_conditioning_scale=cn_scale,
            resolution=res,
            temporal_smoothing=temporal,
        )
        return (out if ok else None), ("✅ 重绘完成！" if ok else "❌ 处理失败")

    except Exception as e:
        logger.exception(f"处理异常: {e}")
        return None, f"❌ 异常: {e}"
    finally:
        with _lock:
            _processing = False


# ════════════════════════════════════════════════════════════
# Gradio 界面
# ════════════════════════════════════════════════════════════

PRESETS = {
    "无预设": ("", ""),
    "真实感增强": (
        "raw photo, ultra realistic, natural lighting, high detail skin texture, "
        "sharp focus, film grain, shot on Canon EOS R5",
        "doll-like, overly smooth, plastic, 3d render, cartoon",
    ),
    "电影质感": (
        "cinematic lighting, color grading, depth of field, film look, "
        "anamorphic lens, professional videography",
        "flat lighting, amateur, oversaturated",
    ),
    "赛博朋克": (
        "cyberpunk aesthetic, neon lights, futuristic, glowing accents, "
        "rain-soaked streets, holographic elements",
        "daylight, natural, plain background",
    ),
    "油画风格": (
        "oil painting style, impressionist, visible brushstrokes, "
        "rich warm colors, gallery art",
        "photorealistic, sharp, digital",
    ),
    "日系动漫": (
        "anime style, cel shading, vibrant colors, detailed eyes, "
        "studio ghibli quality",
        "photorealistic, western, 3d render",
    ),
}


def build_ui():
    with gr.Blocks(
        title="AI 视频智能重绘",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            """
        # 🎬 AI 视频智能重绘工作站
        **上传视频 → 输入提示词 → AI 重绘**　　保留原始内容 · 自然无AI感 · 支持参考图引导
        """
        )

        with gr.Row():
            with gr.Column(scale=2):
                v2v_video = gr.Video(label="📹 输入视频", height=300)
            with gr.Column(scale=2):
                v2v_ref = gr.Image(
                    label="🖼️ 参考图 (可选，引导风格)",
                    type="filepath",
                    height=300,
                )

        with gr.Accordion("⚙️ 提示词与参数", open=True):
            with gr.Row():
                v2v_prompt = gr.Textbox(
                    label="正向提示词 (描述你想要的效果)",
                    placeholder="例如: cinematic lighting, detailed skin, natural hair, warm color grading",
                    lines=3,
                    scale=4,
                )
                v2v_preset = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    label="✨ 预设",
                    value="无预设",
                    scale=1,
                )

            v2v_neg = gr.Textbox(
                label="负向提示词 (排除不想要的)",
                value="low quality, blurry, deformed, ugly, bad anatomy, watermark",
                lines=2,
            )

            with gr.Row():
                v2v_strength = gr.Slider(
                    0.1, 0.9, value=0.35, step=0.05,
                    label="🎨 重绘强度 (越低越接近原视频)",
                )
                v2v_steps = gr.Slider(
                    10, 50, value=25, step=1,
                    label="🔢 渲染步数",
                )
                v2v_res = gr.Slider(
                    512, 1024, value=768, step=64,
                    label="📐 处理分辨率",
                )

            with gr.Row():
                v2v_cfg = gr.Slider(
                    1.0, 20.0, value=7.5, step=0.5,
                    label="📏 引导系数 (CFG)",
                )
                v2v_cn = gr.Slider(
                    0.0, 2.0, value=0.8, step=0.05,
                    label="🔒 结构锁定强度",
                )
                v2v_temporal = gr.Slider(
                    0.0, 0.5, value=0.12, step=0.02,
                    label="🎞️ 时序平滑",
                )

        with gr.Row():
            v2v_btn = gr.Button(
                "🚀 开始重绘", variant="primary", scale=3, size="lg"
            )
            v2v_status = gr.Textbox(label="状态", scale=2, interactive=False)

        v2v_output = gr.Video(label="📤 输出视频", height=400)

        with gr.Accordion("📋 实时日志", open=False):
            v2v_log = gr.Textbox(
                label="",
                lines=15,
                interactive=False,
                autoscroll=True,
            )
            gr.Timer(2.0).tick(fn=get_progress, outputs=v2v_log)

        # 预设回调
        def on_preset(choice, cur_p, cur_n):
            if choice in PRESETS and choice != "无预设":
                p, n = PRESETS[choice]
                new_p = f"{cur_p}, {p}" if cur_p.strip() else p
                new_n = f"{cur_n}, {n}" if cur_n.strip() else n
                return new_p, new_n
            return cur_p, cur_n

        v2v_preset.change(
            fn=on_preset,
            inputs=[v2v_preset, v2v_prompt, v2v_neg],
            outputs=[v2v_prompt, v2v_neg],
        )

        # 主按钮回调
        v2v_btn.click(
            fn=_run_v2v,
            inputs=[
                v2v_video, v2v_ref, v2v_prompt, v2v_neg,
                v2v_strength, v2v_steps, v2v_res, v2v_cfg,
                v2v_cn, v2v_temporal,
            ],
            outputs=[v2v_output, v2v_status],
        )

    return demo


# ════════════════════════════════════════════════════════════
# 启动
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI 视频智能重绘工作站")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--auth",
        nargs=2,
        metavar=("USER", "PASS"),
        help="设置登录认证 (用户名 密码)",
    )
    args = parser.parse_args()

    demo = build_ui()
    launch_kwargs = dict(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
    if args.auth:
        launch_kwargs["auth"] = tuple(args.auth)

    demo.launch(**launch_kwargs)
