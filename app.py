#!/usr/bin/env python3
"""
视频性别转换 - 可视化界面
- Gradio 操作界面（上传/预览/参数控制）
- 国内 AI（DeepSeek/通义千问/文心一言）解析自然语言调整参数
- 实时进度显示

安装:
  pip install gradio openai   # DeepSeek / 通义千问 均兼容 OpenAI SDK

运行:
  python app.py
  python app.py --share              # 生成公网链接（云服务器必须加）
  python app.py --ai deepseek        # 指定 AI 提供商
  python app.py --ai qwen
  python app.py --ai ernie

配置 API Key（任选其一）:
  export DEEPSEEK_API_KEY=sk-xxx     # DeepSeek（推荐，便宜且效果好）
  export DASHSCOPE_API_KEY=sk-xxx    # 通义千问
  export QIANFAN_API_KEY=xxx         # 文心一言
  export MINIMAX_API_KEY=xxx         # MiniMax
"""

import argparse
import json
import logging
import os
import sys
import threading
from pathlib import Path

# ── 导入 gradio，失败则打印修复命令后退出 ──────────────────────
try:
    import gradio as gr
except ImportError as e:
    import importlib.util
    if importlib.util.find_spec('gradio') is None:
        print("[错误] 未安装 gradio，请运行：")
        print("  pip install gradio")
    else:
        print(f"[错误] gradio 导入失败（依赖版本冲突）: {e}")
        print("[修复] 请运行以下命令后重试：")
        print('  pip install "httpx<1.0"')
        print("  pip install --upgrade gradio")
    sys.exit(1)
except Exception as e:
    print(f"[错误] gradio 加载失败: {e}")
    print("[修复] 请运行以下命令后重试：")
    print('  pip install "httpx<1.0"')
    print("  pip install --upgrade gradio")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
# 代理不拦截本地地址（避免 502 Bad Gateway）
os.environ.setdefault('NO_PROXY', 'localhost,127.0.0.1,0.0.0.0')
os.environ.setdefault('no_proxy', 'localhost,127.0.0.1,0.0.0.0')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════
# AI 提供商配置（国内可用）
# ════════════════════════════════════════════════════════════

AI_PROVIDERS = {
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "install": "pip install openai",
    },
    "qwen": {
        "name": "通义千问",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-turbo",
        "env_key": "DASHSCOPE_API_KEY",
        "install": "pip install openai",
    },
    "ernie": {
        "name": "文心一言",
        "base_url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat",
        "model": "ernie-lite-8k",
        "env_key": "QIANFAN_API_KEY",
        "install": "pip install qianfan",
    },
    "zhipu": {
        "name": "智谱 GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
        "env_key": "ZHIPU_API_KEY",
        "install": "pip install openai",
    },
    "minimax": {
        "name": "MiniMax",
        "base_url": "https://api.minimax.chat/v1",
        "model": "abab6.5s-chat",
        "env_key": "MINIMAX_API_KEY",
        "install": "pip install openai",
    },
}

# 当前使用的提供商（可通过命令行参数修改）
_ai_provider = "deepseek"

SYSTEM_PROMPT = """你是一个视频处理参数助手，帮助用户通过自然语言调整视频性别转换的参数。

可调参数说明：
- strength (0.5~1.0)：重绘强度。越高女性化越明显但可能失真，越低保留原始细节但变化小。默认0.85
- guidance_scale (5~12)：AI引导强度。越高越严格遵守提示词，越低更自然。默认7.5
- semitones (1~10)：声音变调半音数。越高声音越尖细，5=适中女声，8=较高女声。默认5
- face_swap (true/false)：是否启用换脸精修。默认true
- prompt_append (字符串)：追加到正面提示词，如"wearing red dress, long black hair"
- neg_append (字符串)：追加到负面提示词

根据用户需求返回 JSON，只包含需要修改的参数：
{
  "strength": 数值,
  "guidance_scale": 数值,
  "semitones": 数值,
  "face_swap": 布尔,
  "prompt_append": "字符串",
  "neg_append": "字符串",
  "explanation": "用中文解释做了哪些调整以及原因"
}
只返回 JSON，不要其他文字。"""


def _call_openai_compatible(provider_cfg: dict, user_input: str,
                             current_params: dict) -> tuple[dict, str]:
    """调用兼容 OpenAI 接口的国内 AI（DeepSeek/通义千问/智谱）"""
    try:
        from openai import OpenAI
    except ImportError:
        return current_params, "⚠️ 缺少依赖，请运行: pip install openai"

    api_key = os.environ.get(provider_cfg['env_key'], '')
    if not api_key:
        return current_params, (
            f"⚠️ 未设置 {provider_cfg['env_key']}，"
            f"请运行: export {provider_cfg['env_key']}=你的密钥"
        )

    client = OpenAI(api_key=api_key, base_url=provider_cfg['base_url'])
    resp = client.chat.completions.create(
        model=provider_cfg['model'],
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"当前参数：{json.dumps(current_params, ensure_ascii=False)}\n\n"
                f"用户需求：{user_input}"
            )},
        ],
    )
    return resp.choices[0].message.content.strip()


def _call_ernie(provider_cfg: dict, user_input: str,
                current_params: dict) -> str:
    """调用文心一言（百度 qianfan SDK）"""
    try:
        import qianfan
    except ImportError:
        raise ImportError("缺少依赖，请运行: pip install qianfan")
    api_key = os.environ.get(provider_cfg['env_key'], '')
    if not api_key:
        raise ValueError(f"未设置 {provider_cfg['env_key']}")

    chat = qianfan.ChatCompletion(ak=api_key)
    resp = chat.do(
        model="ERNIE-Lite-8K",
        messages=[
            {"role": "user", "content": (
                SYSTEM_PROMPT + "\n\n"
                f"当前参数：{json.dumps(current_params, ensure_ascii=False)}\n\n"
                f"用户需求：{user_input}"
            )},
        ],
    )
    return resp['body']['result']


def ai_adjust(user_input: str, current_params: dict) -> tuple[dict, str]:
    """调用国内 AI 解析用户输入，返回（新参数, 说明文字）"""
    provider_cfg = AI_PROVIDERS.get(_ai_provider, AI_PROVIDERS['deepseek'])
    try:
        if _ai_provider == 'ernie':
            raw = _call_ernie(provider_cfg, user_input, current_params)
        else:
            raw = _call_openai_compatible(
                provider_cfg, user_input, current_params
            )
            if isinstance(raw, tuple):   # 未设置 API Key 时直接返回
                return raw

        # 提取 JSON
        if '{' in raw:
            raw = raw[raw.index('{'):raw.rindex('}') + 1]
        result = json.loads(raw)

        explanation = result.pop('explanation', '参数已根据你的需求调整。')
        merged = {**current_params, **result}
        return merged, f"✅ [{provider_cfg['name']}] {explanation}"

    except ImportError:
        cfg = AI_PROVIDERS.get(_ai_provider, AI_PROVIDERS['deepseek'])
        return current_params, (
            f"⚠️ 未安装依赖，请运行: {cfg['install']}"
        )
    except Exception as e:
        return current_params, f"❌ AI 调整失败: {e}"


# ════════════════════════════════════════════════════════════
# 进度收集器
# ════════════════════════════════════════════════════════════

class ProgressCollector(logging.Handler):
    """拦截日志消息，实时推送到 Gradio 界面"""

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


def run_transform(
    video_file,
    face_file,
    strength,
    guidance_scale,
    semitones,
    face_swap,
    skip_body,
    prompt_append,
    neg_append,
    segments,
):
    global _processing
    if _processing:
        return None, "⚠️ 已有任务在运行，请等待完成后再提交"

    if video_file is None:
        return None, "❌ 请先上传输入视频"
    if face_file is None:
        return None, "❌ 请先上传女性人脸图片"

    _processing = True
    _progress.clear()
    output_path = "output_full.mp4"

    try:
        try:
            from run_full_transform import process, process_with_segments
            from src.body_swapper import BodySwapper
        except ImportError as e:
            return None, f"⚠️ 缺少依赖: {e}\n请先运行: pip install -r requirements.txt"

        # 动态注入自定义 prompt
        if prompt_append.strip():
            BodySwapper.DEFAULT_PROMPT = (
                BodySwapper.DEFAULT_PROMPT.rstrip(', ') +
                f", {prompt_append.strip()}"
            )
        if neg_append.strip():
            BodySwapper.DEFAULT_NEG = (
                BodySwapper.DEFAULT_NEG.rstrip(', ') +
                f", {neg_append.strip()}"
            )

        kwargs = dict(
            input_video=video_file,
            female_face=face_file,
            output_path=output_path,
            strength=strength,
            semitones=semitones,
            no_face_swap=not face_swap,
            no_body_swap=skip_body,
        )

        if segments > 1:
            ok = process_with_segments(n_segments=segments, **kwargs)
        else:
            ok = process(**kwargs)

        if ok and Path(output_path).exists():
            return output_path, "✅ 处理完成！"
        else:
            return None, "❌ 处理失败，请查看下方日志"

    except Exception as e:
        logger.exception(f"处理异常: {e}")
        return None, f"❌ 异常: {e}"
    finally:
        _processing = False


def get_progress():
    return _progress.get_log()


# ════════════════════════════════════════════════════════════
# Gradio 界面
# ════════════════════════════════════════════════════════════

def build_ui():
    with gr.Blocks(title="视频性别转换工具") as demo:

        gr.Markdown("""
        # 🎬 视频性别转换工具
        **AnimateDiff + ControlNet** 全身重绘 · **InsightFace** 换脸精修 · **demucs** 声音分离 · **DeepSeek/通义千问** 参数调整
        """)

        # ── 上传区 ──────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                video_input = gr.Video(label="📹 输入视频", height=300)
            with gr.Column(scale=1):
                face_input = gr.Image(
                    label="🖼️ 女性人脸图片（用于换脸精修）",
                    type="filepath",
                    height=300,
                )

        # ── 参数控制 ─────────────────────────────────────────
        with gr.Accordion("⚙️ 处理参数", open=True):
            with gr.Row():
                strength = gr.Slider(
                    0.5, 1.0, value=0.85, step=0.05,
                    label="重绘强度",
                    info="越高女性化越明显，越低保留原始细节"
                )
                guidance_scale = gr.Slider(
                    5.0, 12.0, value=7.5, step=0.5,
                    label="AI引导强度",
                    info="越高越严格遵守风格描述"
                )
                semitones = gr.Slider(
                    1, 10, value=5, step=1,
                    label="声音变调（半音）",
                    info="5=适中女声，8=较高女声"
                )

            with gr.Row():
                face_swap = gr.Checkbox(
                    value=True, label="启用换脸精修（InsightFace）"
                )
                skip_body = gr.Checkbox(
                    value=False, label="⚡ 本地测试模式（跳过全身重绘，仅测试音频+换脸）",
                    info="无GPU时勾选此项，速度快但不重绘身体"
                )
                segments = gr.Slider(
                    1, 8, value=1, step=1,
                    label="视频分段数",
                    info="显存不足时分段处理，推荐1（不分段）"
                )

            with gr.Row():
                prompt_append = gr.Textbox(
                    label="追加描述（正面）",
                    placeholder="例如：wearing elegant dress, long black hair",
                    lines=1,
                )
                neg_append = gr.Textbox(
                    label="追加描述（负面）",
                    placeholder="例如：glasses, short hair",
                    lines=1,
                )

        # ── AI 参数调整 ──────────────────────────────────────
        with gr.Accordion("🤖 AI 自然语言调整参数", open=True):
            gr.Markdown(
                "用自然语言描述你想要的效果，AI 会自动调整上方参数。\n\n"
                "**示例：** `让女生声音更温柔自然` · `画面变化更明显` · "
                "`保留更多原始背景细节` · `换成扎马尾的造型`"
            )
            with gr.Row():
                ai_input = gr.Textbox(
                    label="描述你想要的效果",
                    placeholder="用中文或英文描述…",
                    lines=2,
                    scale=4,
                )
                ai_btn = gr.Button("✨ AI调整", scale=1, variant="secondary")
            ai_output = gr.Textbox(
                label="AI说明", lines=2, interactive=False
            )

        # ── 运行 & 输出 ──────────────────────────────────────
        with gr.Row():
            run_btn = gr.Button(
                "🚀 开始处理", variant="primary", scale=3, size="lg"
            )
            status = gr.Textbox(
                label="状态", scale=2, interactive=False
            )

        output_video = gr.Video(label="📤 输出视频", height=400)

        # ── 实时日志 ─────────────────────────────────────────
        with gr.Accordion("📋 实时日志", open=False):
            log_box = gr.Textbox(
                label="", lines=15, interactive=False,
                elem_classes=["log-box"],
                autoscroll=True,
            )
            gr.Timer(2.0).tick(fn=get_progress, outputs=log_box)

        # ── 事件绑定 ─────────────────────────────────────────

        def _collect_params(s, g, sem, fs, pa, na):
            return {
                "strength": s,
                "guidance_scale": g,
                "semitones": sem,
                "face_swap": fs,
                "prompt_append": pa,
                "neg_append": na,
            }

        def _ai_click(user_input, s, g, sem, fs, pa, na):
            current = _collect_params(s, g, sem, fs, pa, na)
            new_params, explanation = ai_adjust(user_input, current)
            return (
                new_params.get('strength', s),
                new_params.get('guidance_scale', g),
                new_params.get('semitones', sem),
                new_params.get('face_swap', fs),
                new_params.get('prompt_append', pa),
                new_params.get('neg_append', na),
                explanation,
            )

        ai_btn.click(
            fn=_ai_click,
            inputs=[ai_input, strength, guidance_scale, semitones,
                    face_swap, prompt_append, neg_append],
            outputs=[strength, guidance_scale, semitones,
                     face_swap, prompt_append, neg_append, ai_output],
        )

        run_btn.click(
            fn=run_transform,
            inputs=[video_input, face_input, strength, guidance_scale,
                    semitones, face_swap, skip_body, prompt_append, neg_append, segments],
            outputs=[output_video, status],
        )

    return demo


# ════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',   type=int,  default=7860)
    parser.add_argument('--share',  action='store_true',
                        help='生成公网链接（Colab/云服务器必须加此参数）')
    parser.add_argument('--host',   default='127.0.0.1')
    parser.add_argument('--ai',     default='deepseek',
                        choices=list(AI_PROVIDERS.keys()),
                        help='AI提供商: deepseek(默认)/qwen/ernie/zhipu')
    args = parser.parse_args()

    _ai_provider = args.ai
    logger.info(f"AI提供商: {AI_PROVIDERS[_ai_provider]['name']}")

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(),
    )
