# -*- coding: utf-8 -*-
"""
SubtitleLab Translations Module
Provides Chinese (zh_CN) translations for the PyQt6 GUI.
"""

from typing import Dict

# Chinese translations dictionary
ZH_CN: Dict[str, str] = {
    # Main Window
    "SubtitleLab initialized. Ready to import.": "SubtitleLab 已初始化，准备导入文件。",
    "Toggle Theme": "切换主题",
    "Settings": "设置",
    "ORIGINAL SUBTITLES": "原始字幕",
    "PROCESSED OUTPUT": "处理结果",
    "Import File": "导入文件",
    "Start": "开始",
    "Cancel": "取消",
    "Export": "导出",
    "Total": "总计",
    "Processed": "已处理",
    "Time": "用时",
    "SYSTEM LOGS": "系统日志",
    "Import Subtitle File": "导入字幕文件",
    "Subtitle Files (*.srt *.ass *.ssa);;All Files (*)": "字幕文件 (*.srt *.ass *.ssa);;所有文件 (*)",
    "Loading:": "正在加载：",
    "Loaded {count} subtitles": "已加载 {count} 条字幕",
    "Error": "错误",
    "Processing cancelled": "处理已取消",
    "Processing completed successfully": "处理完成",
    "Processing failed": "处理失败",
    "Export Processed Subtitles": "导出处理后的字幕",
    "SRT Files (*.srt);;All Files (*)": "SRT 文件 (*.srt);;所有文件 (*)",
    "Exported to:": "已导出至：",
    "Success": "成功",
    "Successfully exported to:": "成功导出至：",
    "Theme switched to dark": "已切换到深色主题",
    "Theme switched to light": "已切换到浅色主题",
    "Settings saved": "设置已保存",
    # Settings Dialog
    "LLM": "语言模型",
    "Processing": "处理设置",
    "Prompts": "提示词",
    "Pricing": "价格",
    "Preset": "预设",
    "Provider:": "提供商：",
    "API Configuration": "API 配置",
    "API Endpoint:": "API 端点：",
    "API Key:": "API 密钥：",
    "Model:": "模型：",
    "Parameters": "参数",
    "Context Window:": "上下文窗口：",
    "Max Output Tokens:": "最大输出令牌：",
    "Timeout:": "超时时间：",
    "Concurrency:": "并发数：",
    "Enable JSON Mode": "启用 JSON 模式",
    "Chunking Strategy": "分块策略",
    "Window Size:": "窗口大小：",
    "Window Overlap:": "窗口重叠：",
    "Optimization": "优化选项",
    "Enable Semantic Analysis": "启用语义分析",
    "Enable Pre-filter": "启用预过滤",
    "Allow Dynamic Window": "允许动态窗口",
    "Quality Scoring": "质量评分",
    "Enable Quality Scoring (LLM)": "启用质量评分 (LLM)",
    "Score Threshold:": "评分阈值：",
    "Background Information": "背景信息",
    "Context about the video content...": "关于视频内容的背景信息...",
    "Style Guide": "风格指南",
    "Translation tone, specific terminology...": "翻译语气、特定术语...",
    "Custom Instructions": "自定义指令",
    "Any additional prompt instructions...": "任何额外的提示指令...",
    "Token Pricing": "令牌价格",
    "Enable Cost Estimation": "启用成本估算",
    "Input Price:": "输入价格：",
    "Output Price:": "输出价格：",
    "Used for calculating estimated costs based on token usage.": "用于根据令牌使用量计算预估成本。",
    "Save": "保存",
    # Components
    "ORIGINAL": "原文",
    "PROCESSED": "处理后",
    # Actions
    "merge": "合并",
    "delete": "删除",
    "correct": "修正",
    "keep": "保留",
    "MERGE": "合并",
    "DELETE": "删除",
    "CORRECT": "修正",
    "KEEP": "保留",
}

# English (default) - passthrough
EN_US: Dict[str, str] = {}


def get_translation(key: str, lang: str = "zh_CN") -> str:
    """
    Get translation for a key.

    Args:
        key: The text to translate
        lang: Language code (zh_CN, en_US)

    Returns:
        Translated text or original key if not found
    """
    if lang == "zh_CN":
        return ZH_CN.get(key, key)
    return key


def get_translations(lang: str = "zh_CN") -> Dict[str, str]:
    """
    Get all translations for a language.

    Args:
        lang: Language code

    Returns:
        Dictionary of translations
    """
    if lang == "zh_CN":
        return ZH_CN
    return EN_US
