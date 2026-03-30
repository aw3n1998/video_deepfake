"""
提示词意图路由 (Prompt Router)

根据提示词内容判断应该走哪条管线：
- "style"   → 现有 SD1.5 img2img 风格重绘
- "content" → Wan2.1 视频内容生成（动作/物体/场景级改变）

设计原则：纯规则 + 关键词打分，无需额外模型，延迟 <1ms。
"""

import re
from typing import Literal, Optional

# ════════════════════════════════════════════════════════════
# 关键词库
# ════════════════════════════════════════════════════════════

# 内容/动作变更关键词（表明用户想改变"发生了什么"）
_CONTENT_KEYWORDS = [
    # 中文 — 动作/变化
    "变成", "改成", "换成", "替换", "增加", "添加", "删除", "去掉", "移除",
    "让人物", "让他", "让她", "让它", "出现", "消失", "拿着", "拿起", "放下",
    "走路", "跑步", "跳舞", "转身", "回头", "挥手", "梳头", "梳头发",
    "掉落", "飞起", "下雨", "下雪", "爆炸", "燃烧", "开花",
    "穿上", "脱下", "戴上", "摘下",
    "场景变", "背景变", "环境变",
    # 中文 — 物体/主体
    "加一个", "加个", "多一个", "多个",
    "梳子", "伞", "花", "刀", "剑", "枪", "帽子", "眼镜",
    # 英文
    "turn into", "change to", "replace with", "add a", "remove the",
    "make the person", "make him", "make her", "make it",
    "walking", "running", "dancing", "jumping", "waving",
    "appear", "disappear", "holding", "wearing",
    "scene change", "background change",
]

# 风格变更关键词（表明用户想改变"画面看起来怎样"）
_STYLE_KEYWORDS = [
    # 中文
    "风格", "质感", "色调", "光照", "光影", "滤镜", "画风",
    "赛博朋克", "油画", "水彩", "素描", "动漫", "卡通", "写实",
    "电影", "复古", "梦幻", "黑白", "暖色", "冷色", "高对比",
    "柔光", "硬光", "逆光", "日系", "欧美", "中国风",
    "真实感", "增强", "清晰", "模糊", "柔化", "锐化",
    # 英文
    "style", "cinematic", "anime", "realistic", "painting",
    "vintage", "cyberpunk", "fantasy", "watercolor", "cartoon",
    "black and white", "warm tone", "cool tone", "film grain",
    "hdr", "low key", "high key", "bokeh",
]


def classify_prompt(prompt: str) -> Literal["style", "content"]:
    """
    分析提示词意图。

    打分规则：
    - 匹配到内容关键词 → +1 content 分
    - 匹配到风格关键词 → +1 style 分
    - 得分高的胜出；平局或都为 0 → 默认 "style"（保守路由到现有管线）

    Returns:
        "style" 或 "content"
    """
    if not prompt or not prompt.strip():
        return "style"

    prompt_lower = prompt.lower()

    content_score = sum(1 for kw in _CONTENT_KEYWORDS if kw in prompt_lower)
    style_score = sum(1 for kw in _STYLE_KEYWORDS if kw in prompt_lower)

    if content_score > style_score:
        return "content"
    return "style"


def route_pipeline(
    prompt: str, reference_image: Optional[str] = None
) -> Literal["vid2vid", "vid2vid_gen"]:
    """
    综合判断应该使用哪条管线。

    Args:
        prompt: 用户提示词
        reference_image: 参考图路径（有参考图时倾向风格迁移）

    Returns:
        "vid2vid"     → SD1.5 img2img 风格重绘
        "vid2vid_gen" → Wan2.1 内容生成
    """
    intent = classify_prompt(prompt)

    # 有参考图且意图不明确时，倾向风格迁移（参考图是风格引导的典型用法）
    if reference_image and intent == "style":
        return "vid2vid"

    if intent == "content":
        return "vid2vid_gen"

    return "vid2vid"
