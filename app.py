#!/usr/bin/env python3
"""
AI 视频重绘工作站 - 可视化界面
针对 RTX 5090 优化，支持高质量通用视频重绘 (Vid2Vid) 与 参考图引导。
"""

import argparse
import json
import logging
import os
import sys
import threading
from pathlib import Path

try:
    import gradio as gr
except ImportError:
    print("[错误] 未安装 gradio")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('NO_PROXY', 'localhost,127.0.0.1,0.0.0.0')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════
# AI 提供商配置
# ════════════════════════════════════════════════════════════

AI_PROVIDERS = {
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "name": "通义千问",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-turbo",
        "env_key": "DASHSCOPE_API_KEY",
    },
}

_ai_provider = "deepseek"

SYSTEM_PROMPT = """你是一个视频处理助手。用户想要通过自然语言修改视频内容。
请将用户的描述转化为合适的正向提示词 (Prompt) 和参数。
特别注意：如果用户提到“素人”、“真实”、“梳头”、“掉发”，请生成强调皮肤质感和物理细节的词。
"""

def ai_adjust(user_input: str, current_params: dict) -> tuple[dict, str]:
    # 简化版实现，实际调用逻辑可按需恢复
    return current_params, "AI 建议：使用真实感预设以获得最佳效果。"

# ════════════════════════════════════════════════════════════
# 进度收集器
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
            if len(self.lines) > 200:
                self.lines = self.lines[-200:]

    def get_log(self) -> str:
        with self._lock:
            return '\n'.join(self.lines)

    def clear(self):
        with self._lock:
            self.lines.clear()

_progress = ProgressCollector()
_progress.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S'))
logging.getLogger().addHandler(_progress)

# ════════════════════════════════════════════════════════════
# 核心处理函数
# ════════════════════════════════════════════════════════════

_processing = False

def run_legacy_transform(video, face):
    global _processing
    if _processing: return None, "⚠️ 任务运行中..."
    if not video or not face: return None, "❌ 缺少输入"
    
    _processing = True
    _progress.clear()
    out = "output_legacy.mp4"
    try:
        from run_full_transform import process
        ok = process(input_video=video, target_face=face, output_path=out, strength=0.85)
        return out if ok else None, "完成" if ok else "失败"
    except Exception as e:
        return None, f"错误: {e}"
    finally:
        _processing = False

def get_progress():
    return _progress.get_log()

# ════════════════════════════════════════════════════════════
# Gradio 界面
# ════════════════════════════════════════════════════════════

def build_ui():
    with gr.Blocks(title="AI 视频高清重绘工作站") as demo:

        gr.Markdown("""
        # 🎬 AI 视频高清重绘工作站 (RTX 5090 专供)
        **AnimateDiff + Dual ControlNet** 全场景重绘 · **无硬编码** 全自定义转换 · **素人真实感** 优化
        """)
        
        with gr.Tabs():
            # ────────────────────────────────────────────────────────
            # 标签页 1: 通用高质量视频重绘 (Vid2Vid)
            # ────────────────────────────────────────────────────────
            with gr.Tab("🚀 通用高质量视频重绘"):
                gr.Markdown("支持上传视频和参考图，通过提示词修改任何内容。针对“梳头、掉发”等物理动作做了深度连贯性优化。")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        v2v_video_input = gr.Video(label="📹 输入原视频", height=300)
                    with gr.Column(scale=2):
                        v2v_ref_img = gr.Image(label="🖼️ 参考图/目标人物 (可选)", type="filepath", height=300)
                        
                with gr.Accordion("⚙️ 风格与参数控制", open=True):
                    with gr.Row():
                        v2v_prompt = gr.Textbox(
                            label="修改内容描述 (Prompt)", 
                            placeholder="例如：raw photo, a person combing hair, hair falling out, strands on comb, messy real hair", 
                            lines=3,
                            scale=4
                        )
                        with gr.Column(scale=1):
                            v2v_preset = gr.Dropdown(
                                choices=["无预设", "真实感日常 (素人)", "梳头掉发反馈", "赛博朋克风格", "日系动漫"],
                                label="✨ 效果预设",
                                value="无预设"
                            )
                    
                    v2v_neg_prompt = gr.Textbox(
                        label="负向提示词", 
                        value="low quality, low res, blurry, doll-like, fake, 3d render, anime style", 
                        lines=2
                    )
                    
                    with gr.Row():
                        v2v_strength = gr.Slider(0.1, 1.0, value=0.75, step=0.05, label="重绘强度 (Strength)")
                        v2v_steps = gr.Slider(10, 60, value=30, step=1, label="渲染步数")
                        v2v_resolution = gr.Slider(512, 1024, value=768, step=8, label="输出分辨率")
                        v2v_guidance = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="引导系数 (CFG)")
                        
                    with gr.Row():
                        v2v_pose_weight = gr.Slider(0.0, 2.0, value=0.85, step=0.05, label="动作锁定 (Pose)")
                        v2v_depth_weight = gr.Slider(0.0, 2.0, value=0.75, step=0.05, label="结构锁定 (Depth)")

                with gr.Row():
                    v2v_run_btn = gr.Button("🚀 立即开始重绘", variant="primary", scale=3, size="lg")
                    v2v_status = gr.Textbox(label="状态", scale=2, interactive=False)
                    
                v2v_output_video = gr.Video(label="📤 重绘结果", height=400)
                
                with gr.Accordion("📋 实时渲染日志", open=False):
                    v2v_log_box = gr.Textbox(label="", lines=12, interactive=False, elem_classes=["log-box"], autoscroll=True)
                    gr.Timer(2.0).tick(fn=get_progress, outputs=v2v_log_box)

                def on_preset_change(choice, current_p, current_n):
                    presets = {
                        "真实感日常 (素人)": ("raw photo, shot on phone, amateur video, natural lighting, high detail skin", "doll-like, overly polished, 3d"),
                        "梳头掉发反馈": ("realistic hair texture, loose hair strands falling, strands on comb, messy real hair", "bald, fake hair"),
                        "赛博朋克风格": ("cyberpunk theme, neon lights, glowing accents", "low res"),
                        "日系动漫": ("anime style, vibrant colors", "photorealistic")
                    }
                    if choice in presets:
                        p, n = presets[choice]
                        return f"{current_p}, {p}" if current_p else p, f"{current_n}, {n}" if current_n else n
                    return current_p, current_n

                v2v_preset.change(fn=on_preset_change, inputs=[v2v_preset, v2v_prompt, v2v_neg_prompt], outputs=[v2v_prompt, v2v_neg_prompt])

                def _run_v2v(video, ref, prompt, neg, strength, steps, res, cfg, pose_w, depth_w):
                    global _processing
                    if _processing: return None, "⚠️ 任务队列中..."
                    if not video: return None, "❌ 请选择视频"
                    _processing = True
                    _progress.clear()
                    out = "output_v2v.mp4"
                    try:
                        from src.vid2vid import Vid2VidPipeline
                        pipe = Vid2VidPipeline(device="cuda")
                        ok = pipe.process_video(
                            video_path=video, output_path=out, prompt=prompt, negative_prompt=neg,
                            reference_image=ref, num_steps=steps, strength=strength,
                            guidance_scale=cfg, controlnet_conditioning_scale=[pose_w, depth_w], resolution=res
                        )
                        return out if ok else None, "✅ 成功" if ok else "❌ 失败"
                    except Exception as e:
                        return None, f"❌ 异常: {e}"
                    finally:
                        _processing = False

                v2v_run_btn.click(
                    fn=_run_v2v,
                    inputs=[v2v_video_input, v2v_ref_img, v2v_prompt, v2v_neg_prompt, v2v_strength, v2v_steps, v2v_resolution, v2v_guidance, v2v_pose_weight, v2v_depth_weight],
                    outputs=[v2v_output_video, v2v_status]
                )

            # ────────────────────────────────────────────────────────
            # 标签页 2: 原版功能入口 (备用)
            # ────────────────────────────────────────────────────────
            with gr.Tab("🔧 智能人像与声音转换 (备用)"):
                gr.Markdown("此页提供基于人脸替换与声音变调的快速转换逻辑。")
                with gr.Row():
                    v_in = gr.Video(label="输入视频")
                    f_in = gr.Image(label="目标人脸/风格图", type="filepath")
                b_run = gr.Button("启动转换")
                o_v = gr.Video(label="输出结果")
                b_run.click(fn=run_legacy_transform, inputs=[v_in, f_in], outputs=[o_v, gr.State("")])

    return demo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
