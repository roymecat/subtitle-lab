"""
Semantic analyzer module for SubtitleLab.

Uses LLM to pre-analyze subtitle content for better context understanding.
Enhanced with deep semantic analysis and adaptive sampling.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .models import SemanticContext, SubtitleEntry

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)


# =============================================================================
# Enhanced Analysis Prompt V2
# =============================================================================

ANALYSIS_PROMPT_V2 = """<task>深度分析字幕内容，提取用于智能校对的语义信息。</task>

<input>
{subtitle_sample}
</input>

<analysis_dimensions>
1. **内容理解**
   - 这是什么类型的内容？(电影/电视剧/纪录片/访谈/教程/动漫/其他)
   - 主要讲述什么？(1-2句核心内容)

2. **人物分析**
   - 识别所有出现的人物/角色名称
   - 注意人物的说话特点（如有）

3. **语言特征**
   - 整体语言风格（正式/口语/专业/幽默）
   - 是否有方言/口音特征
   - 专业术语/专有名词列表
   - 常见的语气词使用模式

4. **对话模式**
   - 是独白为主还是对话为主？
   - 说话人切换的典型模式

5. **Whisper识别特征**
   - 观察到的典型识别错误模式
   - 容易混淆的词汇对
   - 断句问题的特征
</analysis_dimensions>

<output_schema>
{{
  "content_type": "电影|电视剧|纪录片|访谈|教程|动漫|其他",
  "summary": "核心内容概述(50字内)",
  "characters": ["人物名称列表"],
  "dialogue_pattern": "独白为主|对话为主|混合",
  "language_style": {{
    "formality": "正式|半正式|口语|混合",
    "special_features": ["方言特征", "专业领域等"],
    "common_fillers": ["常见语气词"]
  }},
  "terminology": ["专有名词/术语列表"],
  "whisper_patterns": {{
    "common_errors": ["观察到的错误模式"],
    "confusable_words": [["词A", "词B"]],
    "segmentation_issues": "断句问题描述"
  }},
  "processing_hints": ["给校对员的建议"]
}}
</output_schema>

<rules>
- 基于实际观察，不要臆测
- 无法确定的字段用空字符串或空数组
- 专注于对后续校对有帮助的信息
- 只输出JSON，无其他内容
</rules>"""


# Legacy prompt for backward compatibility
ANALYSIS_PROMPT_V1 = """<task>分析字幕样本，提取语义信息用于后续校对。</task>

<input>
{subtitle_sample}
</input>

<output_schema>
{{
  "summary": "1-2句内容概述",
  "characters": ["人物名称"],
  "topics": ["主题"],
  "style": "正式|非正式|技术|口语",
  "language_notes": "方言/术语/特殊表达"
}}
</output_schema>

<rules>
- 只输出JSON
- 无法确定的字段用空字符串或空列表
- 人物名称只包含明确提到的
- 每字段不超过100字
</rules>"""


# =============================================================================
# Adaptive Sampling Configuration
# =============================================================================


@dataclass
class SamplingConfig:
    """Configuration for adaptive sampling strategy."""

    # Base sampling counts
    first_n: int = 15
    middle_n: int = 20
    last_n: int = 10

    # Adaptive scaling factors
    min_total_samples: int = 25
    max_total_samples: int = 80

    # Density-based sampling
    enable_density_sampling: bool = True
    dense_region_threshold: float = 0.5  # seconds - gaps smaller than this indicate dense dialogue

    # Scene change detection
    enable_scene_detection: bool = True
    scene_gap_threshold: float = 3.0  # seconds - gaps larger than this suggest scene change


# =============================================================================
# Sampling Functions
# =============================================================================


def _calculate_adaptive_sample_size(
    total_entries: int,
    config: SamplingConfig,
) -> tuple[int, int, int]:
    """
    Calculate adaptive sample sizes based on total entries.

    For larger files, we need more samples to capture the full context.
    Uses logarithmic scaling to avoid excessive sampling.

    Args:
        total_entries: Total number of subtitle entries
        config: Sampling configuration

    Returns:
        Tuple of (first_n, middle_n, last_n)
    """
    if total_entries <= config.min_total_samples:
        return total_entries, 0, 0  # Return all entries

    # Logarithmic scaling: more entries = more samples, but diminishing returns
    scale_factor = 1.0 + math.log10(max(total_entries / 100, 1)) * 0.5
    scale_factor = min(scale_factor, 2.5)  # Cap at 2.5x

    first_n = min(int(config.first_n * scale_factor), total_entries // 4)
    middle_n = min(int(config.middle_n * scale_factor), total_entries // 2)
    last_n = min(int(config.last_n * scale_factor), total_entries // 4)

    # Ensure we don't exceed max samples
    total_samples = first_n + middle_n + last_n
    if total_samples > config.max_total_samples:
        ratio = config.max_total_samples / total_samples
        first_n = int(first_n * ratio)
        middle_n = int(middle_n * ratio)
        last_n = int(last_n * ratio)

    return first_n, middle_n, last_n


def _detect_scene_boundaries(
    entries: list[SubtitleEntry],
    gap_threshold: float = 3.0,
) -> list[int]:
    """
    Detect potential scene boundaries based on time gaps.

    Args:
        entries: List of subtitle entries
        gap_threshold: Minimum gap (seconds) to consider as scene boundary

    Returns:
        List of indices where scene changes likely occur
    """
    boundaries = []

    for i in range(1, len(entries)):
        gap = entries[i].start - entries[i - 1].end
        if gap >= gap_threshold:
            boundaries.append(i)

    return boundaries


def _sample_around_boundaries(
    entries: list[SubtitleEntry],
    boundaries: list[int],
    samples_per_boundary: int = 2,
) -> list[SubtitleEntry]:
    """
    Sample entries around detected scene boundaries.

    Args:
        entries: All subtitle entries
        boundaries: Indices of scene boundaries
        samples_per_boundary: Number of entries to sample around each boundary

    Returns:
        List of sampled entries
    """
    sampled_indices = set()

    for boundary_idx in boundaries:
        # Sample before boundary
        for offset in range(samples_per_boundary):
            idx = boundary_idx - 1 - offset
            if idx >= 0:
                sampled_indices.add(idx)

        # Sample after boundary
        for offset in range(samples_per_boundary):
            idx = boundary_idx + offset
            if idx < len(entries):
                sampled_indices.add(idx)

    return [entries[i] for i in sorted(sampled_indices)]


def _sample_subtitles_adaptive(
    entries: list[SubtitleEntry],
    config: Optional[SamplingConfig] = None,
) -> list[SubtitleEntry]:
    """
    Adaptive sampling strategy that considers content density and scene changes.

    Args:
        entries: All subtitle entries
        config: Sampling configuration (uses defaults if None)

    Returns:
        List of sampled subtitle entries
    """
    if not entries:
        return []

    config = config or SamplingConfig()
    total = len(entries)

    # Calculate adaptive sample sizes
    first_n, middle_n, last_n = _calculate_adaptive_sample_size(total, config)

    # If total is small enough, return all
    if total <= first_n + middle_n + last_n:
        return entries.copy()

    sampled: list[SubtitleEntry] = []
    sampled_ids: set[int] = set()

    # 1. First N entries (opening context)
    for entry in entries[:first_n]:
        if entry.id not in sampled_ids:
            sampled.append(entry)
            sampled_ids.add(entry.id)

    # 2. Scene boundary samples (if enabled)
    if config.enable_scene_detection:
        boundaries = _detect_scene_boundaries(entries, config.scene_gap_threshold)
        boundary_samples = _sample_around_boundaries(entries, boundaries, 2)
        for entry in boundary_samples:
            if entry.id not in sampled_ids:
                sampled.append(entry)
                sampled_ids.add(entry.id)

    # 3. Middle entries (evenly distributed)
    middle_start = first_n
    middle_end = total - last_n

    if middle_end > middle_start and middle_n > 0:
        # Adjust middle_n based on how many we've already sampled
        remaining_middle = middle_n - len(
            [s for s in sampled if middle_start <= entries.index(s) < middle_end if s in entries]
        )
        remaining_middle = max(remaining_middle, middle_n // 2)

        middle_range = middle_end - middle_start
        step = middle_range / (remaining_middle + 1)

        for i in range(1, remaining_middle + 1):
            idx = int(middle_start + i * step)
            if idx < middle_end and entries[idx].id not in sampled_ids:
                sampled.append(entries[idx])
                sampled_ids.add(entries[idx].id)

    # 4. Last N entries (ending context)
    for entry in entries[-last_n:]:
        if entry.id not in sampled_ids:
            sampled.append(entry)
            sampled_ids.add(entry.id)

    # Sort by original order
    sampled.sort(key=lambda e: e.id)

    return sampled


def _sample_subtitles_simple(
    entries: list[SubtitleEntry],
    first_n: int = 10,
    middle_n: int = 10,
    last_n: int = 5,
) -> list[SubtitleEntry]:
    """
    Simple sampling strategy (legacy, for backward compatibility).

    Takes entries from the beginning, middle (evenly distributed), and end.

    Args:
        entries: All subtitle entries
        first_n: Number of entries from the beginning
        middle_n: Number of evenly distributed entries from the middle
        last_n: Number of entries from the end

    Returns:
        List of sampled subtitle entries
    """
    if not entries:
        return []

    total = len(entries)

    if total <= first_n + middle_n + last_n:
        return entries.copy()

    sampled: list[SubtitleEntry] = []

    # First N entries
    sampled.extend(entries[:first_n])

    # Middle entries (evenly distributed)
    middle_start = first_n
    middle_end = total - last_n

    if middle_end > middle_start and middle_n > 0:
        middle_range = middle_end - middle_start
        step = middle_range / (middle_n + 1)

        for i in range(1, middle_n + 1):
            idx = int(middle_start + i * step)
            if idx < middle_end:
                sampled.append(entries[idx])

    # Last N entries
    sampled.extend(entries[-last_n:])

    return sampled


def _format_sample_for_prompt(
    entries: list[SubtitleEntry],
    include_timing: bool = True,
) -> str:
    """
    Format sampled entries for inclusion in the analysis prompt.

    Args:
        entries: Sampled subtitle entries
        include_timing: Whether to include timing information

    Returns:
        Formatted string representation
    """
    lines = []
    for entry in entries:
        text = entry.text.replace("\n", " ")
        if include_timing:
            lines.append(f"[{entry.id}] ({entry.start:.1f}-{entry.end:.1f}s) {text}")
        else:
            lines.append(f"[{entry.id}] {text}")

    return "\n".join(lines)


def _parse_analysis_response_v2(response_data: dict) -> SemanticContext:
    """
    Parse V2 analysis response into SemanticContext.

    Args:
        response_data: Parsed JSON response

    Returns:
        SemanticContext with extracted information
    """
    # Extract language style info
    lang_style = response_data.get("language_style", {})
    style_parts = []

    if lang_style.get("formality"):
        style_parts.append(lang_style["formality"])
    if lang_style.get("special_features"):
        style_parts.extend(lang_style["special_features"])

    style = ", ".join(style_parts) if style_parts else response_data.get("content_type", "")

    # Build language notes from whisper patterns
    whisper_patterns = response_data.get("whisper_patterns", {})
    language_notes_parts = []

    if whisper_patterns.get("common_errors"):
        language_notes_parts.append(f"常见错误: {', '.join(whisper_patterns['common_errors'])}")
    if whisper_patterns.get("confusable_words"):
        confusables = [
            f"{pair[0]}↔{pair[1]}"
            for pair in whisper_patterns["confusable_words"]
            if len(pair) >= 2
        ]
        if confusables:
            language_notes_parts.append(f"易混淆: {', '.join(confusables)}")
    if whisper_patterns.get("segmentation_issues"):
        language_notes_parts.append(f"断句: {whisper_patterns['segmentation_issues']}")

    # Add processing hints
    hints = response_data.get("processing_hints", [])
    if hints:
        language_notes_parts.append(f"建议: {'; '.join(hints)}")

    language_notes = " | ".join(language_notes_parts)

    # Extract topics from terminology and content type
    topics = response_data.get("terminology", [])
    if response_data.get("content_type"):
        topics = [response_data["content_type"]] + topics

    return SemanticContext(
        summary=response_data.get("summary", ""),
        characters=response_data.get("characters", []),
        topics=topics,
        style=style,
        language_notes=language_notes,
    )


def _parse_analysis_response_v1(response: str) -> SemanticContext:
    """
    Parse V1 analysis response (legacy format).

    Args:
        response: Raw LLM response string

    Returns:
        Parsed SemanticContext
    """
    response = response.strip()

    # Handle markdown code blocks
    if response.startswith("```"):
        lines = response.split("\n")
        start_idx = 0
        end_idx = len(lines)

        for i, line in enumerate(lines):
            if line.startswith("```") and i == 0:
                start_idx = 1
            elif line.startswith("```"):
                end_idx = i
                break

        response = "\n".join(lines[start_idx:end_idx])

    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse analysis response as JSON: {e}")
        raise ValueError(f"Invalid JSON response: {e}") from e

    return SemanticContext(
        summary=data.get("summary", ""),
        characters=data.get("characters", []),
        topics=data.get("topics", []),
        style=data.get("style", ""),
        language_notes=data.get("language_notes", ""),
    )


# =============================================================================
# Main Analyzer Class
# =============================================================================


class SemanticAnalyzer:
    """
    Analyzes subtitle content using LLM for semantic understanding.

    This analyzer samples subtitle entries and uses an LLM to extract
    semantic information like content summary, character names, topics,
    speaking style, and language-specific notes.

    Enhanced with:
    - Adaptive sampling based on content size
    - Scene boundary detection
    - Deep semantic analysis prompt
    """

    def __init__(
        self,
        llm_client: "LLMClient",
        sampling_config: Optional[SamplingConfig] = None,
        use_v2_prompt: bool = True,
    ):
        """
        Initialize the semantic analyzer.

        Args:
            llm_client: LLM client for API calls
            sampling_config: Configuration for adaptive sampling
            use_v2_prompt: Whether to use enhanced V2 prompt (default True)
        """
        self.llm_client = llm_client
        self.sampling_config = sampling_config or SamplingConfig()
        self.use_v2_prompt = use_v2_prompt

        # Legacy compatibility
        self.first_n = self.sampling_config.first_n
        self.middle_n = self.sampling_config.middle_n
        self.last_n = self.sampling_config.last_n

    async def analyze(self, entries: list[SubtitleEntry]) -> SemanticContext:
        """
        Analyze subtitle entries and extract semantic context.

        Args:
            entries: List of subtitle entries to analyze

        Returns:
            SemanticContext with extracted information
        """
        if not entries:
            logger.warning("No entries provided for semantic analysis")
            return SemanticContext()

        try:
            # Use adaptive sampling
            sampled = _sample_subtitles_adaptive(entries, self.sampling_config)

            logger.info(
                f"Semantic analysis: sampled {len(sampled)} entries from {len(entries)} total"
            )

            # Format for prompt (include timing for V2)
            sample_text = _format_sample_for_prompt(sampled, include_timing=self.use_v2_prompt)

            # Select prompt version
            prompt_template = ANALYSIS_PROMPT_V2 if self.use_v2_prompt else ANALYSIS_PROMPT_V1
            prompt = prompt_template.format(subtitle_sample=sample_text)

            system_content = (
                "你是一个专业的字幕内容分析助手。请深入分析提供的字幕样本，"
                "提取对后续校对工作有帮助的语义信息。注意识别Whisper语音识别的典型错误模式。"
                if self.use_v2_prompt
                else "你是一个字幕内容分析助手。请分析提供的字幕样本并提取语义信息。"
            )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]

            response = await self.llm_client.chat_completion(
                messages=messages,
                json_mode=True,
            )

            # Parse response based on format
            if isinstance(response.content, dict):
                if self.use_v2_prompt:
                    context = _parse_analysis_response_v2(response.content)
                else:
                    context = SemanticContext(
                        summary=response.content.get("summary", ""),
                        characters=response.content.get("characters", []),
                        topics=response.content.get("topics", []),
                        style=response.content.get("style", ""),
                        language_notes=response.content.get("language_notes", ""),
                    )
            else:
                context = _parse_analysis_response_v1(response.raw_content)

            logger.info(
                f"Semantic analysis complete: "
                f"summary={len(context.summary)} chars, "
                f"characters={len(context.characters)}, "
                f"topics={len(context.topics)}"
            )

            return context

        except Exception as e:
            logger.warning(
                f"Semantic analysis failed, returning empty context: {e}",
                exc_info=True,
            )
            return SemanticContext()

    async def analyze_with_custom_prompt(
        self,
        entries: list[SubtitleEntry],
        custom_prompt: str,
    ) -> SemanticContext:
        """
        Analyze subtitle entries with a custom analysis prompt.

        Args:
            entries: List of subtitle entries to analyze
            custom_prompt: Custom prompt template (must contain {subtitle_sample})

        Returns:
            SemanticContext with extracted information
        """
        if not entries:
            logger.warning("No entries provided for semantic analysis")
            return SemanticContext()

        if "{subtitle_sample}" not in custom_prompt:
            raise ValueError("Custom prompt must contain {subtitle_sample} placeholder")

        try:
            # Use adaptive sampling
            sampled = _sample_subtitles_adaptive(entries, self.sampling_config)

            # Format for prompt
            sample_text = _format_sample_for_prompt(sampled, include_timing=True)
            prompt = custom_prompt.format(subtitle_sample=sample_text)

            messages = [
                {"role": "system", "content": "你是一个字幕内容分析助手。"},
                {"role": "user", "content": prompt},
            ]

            response = await self.llm_client.chat_completion(
                messages=messages,
                json_mode=True,
            )

            if isinstance(response.content, dict):
                return SemanticContext(
                    summary=response.content.get("summary", ""),
                    characters=response.content.get("characters", []),
                    topics=response.content.get("topics", []),
                    style=response.content.get("style", ""),
                    language_notes=response.content.get("language_notes", ""),
                )
            return _parse_analysis_response_v1(response.raw_content)

        except Exception as e:
            logger.warning(
                f"Custom semantic analysis failed, returning empty context: {e}",
                exc_info=True,
            )
            return SemanticContext()


# =============================================================================
# Factory Functions
# =============================================================================


def create_semantic_analyzer(
    llm_client: "LLMClient",
    enhanced: bool = True,
) -> SemanticAnalyzer:
    """
    Create a semantic analyzer with recommended settings.

    Args:
        llm_client: LLM client for API calls
        enhanced: Whether to use enhanced V2 features (default True)

    Returns:
        Configured SemanticAnalyzer instance
    """
    config = SamplingConfig(
        first_n=15,
        middle_n=20,
        last_n=10,
        enable_density_sampling=enhanced,
        enable_scene_detection=enhanced,
    )

    return SemanticAnalyzer(
        llm_client=llm_client,
        sampling_config=config,
        use_v2_prompt=enhanced,
    )
