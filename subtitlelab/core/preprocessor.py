"""
Preprocessor module for SubtitleLab.

Implements hallucination detection and rule-based pre-filtering for
Whisper-generated subtitles. Uses conservative rules to avoid false positives.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .models import SubtitleEntry, ProcessedEntry, ProcessingAction


class HallucinationType(str, Enum):
    """Types of hallucinations that can be detected."""

    REPETITIVE_CHARS = "repetitive_chars"
    COMMON_HALLUCINATION = "common_hallucination"
    SINGLE_MEANINGLESS = "single_meaningless"
    BRACKETED_ANNOTATION = "bracketed_annotation"
    JAPANESE_HALLUCINATION = "japanese_hallucination"
    EMPTY_OR_WHITESPACE = "empty_or_whitespace"


@dataclass
class DetectionResult:
    """
    Result of hallucination detection for a single entry.

    Attributes:
        is_hallucination: Whether the entry is detected as a hallucination
        hallucination_type: Type of hallucination detected (if any)
        confidence: Confidence score (0.0 to 1.0)
        reason: Human-readable explanation
        matched_pattern: The pattern or text that triggered detection
    """

    is_hallucination: bool
    hallucination_type: Optional[HallucinationType] = None
    confidence: float = 0.0
    reason: str = ""
    matched_pattern: str = ""

    @property
    def should_filter(self) -> bool:
        """
        Determine if this entry should be filtered based on detection.

        Only filter if confidence is high enough (>= 0.8).
        """
        return self.is_hallucination and self.confidence >= 0.8


@dataclass
class FilteredEntry:
    """
    An entry that was filtered out during preprocessing.

    Attributes:
        entry: The original subtitle entry
        detection: The detection result that caused filtering
    """

    entry: SubtitleEntry
    detection: DetectionResult


@dataclass
class FilterStats:
    """
    Statistics about the filtering process.

    Attributes:
        total_entries: Total number of entries processed
        filtered_count: Number of entries filtered out
        by_type: Count of filtered entries by hallucination type
        by_confidence: Distribution of confidence scores
    """

    total_entries: int = 0
    filtered_count: int = 0
    by_type: dict[HallucinationType, int] = field(default_factory=dict)
    kept_count: int = 0

    @property
    def filter_rate(self) -> float:
        """Percentage of entries that were filtered."""
        if self.total_entries == 0:
            return 0.0
        return (self.filtered_count / self.total_entries) * 100

    def add_filtered(self, hallucination_type: HallucinationType) -> None:
        """Record a filtered entry."""
        self.filtered_count += 1
        self.by_type[hallucination_type] = self.by_type.get(hallucination_type, 0) + 1

    def add_kept(self) -> None:
        """Record a kept entry."""
        self.kept_count += 1


@dataclass
class PreFilterResult:
    """
    Result of the pre-filtering process.

    Attributes:
        kept_entries: Entries that passed filtering
        filtered_entries: Entries that were filtered out with reasons
        stats: Statistics about the filtering process
    """

    kept_entries: list[SubtitleEntry] = field(default_factory=list)
    filtered_entries: list[FilteredEntry] = field(default_factory=list)
    stats: FilterStats = field(default_factory=FilterStats)


class HallucinationDetector:
    """
    Detects hallucinations in Whisper-generated subtitles.

    Uses conservative rules to minimize false positives while catching
    common Whisper hallucination patterns.

    Rules are designed to be CONSERVATIVE:
    - Only flag content that is CLEARLY problematic
    - Repetition threshold is 5+ consecutive same characters
    - Single characters only flagged if pure interjections AND very short duration
    - All detections include confidence scores
    """

    # Minimum consecutive repetitions to trigger detection
    REPETITION_THRESHOLD = 5

    # Maximum duration (seconds) for single-char interjection filtering
    SINGLE_CHAR_MAX_DURATION = 0.5

    # Common Whisper hallucinations in Chinese
    CHINESE_HALLUCINATIONS = [
        "感谢观看",
        "谢谢观看",
        "订阅频道",
        "请订阅",
        "点赞订阅",
        "关注我们",
        "感谢收看",
        "谢谢收看",
        "欢迎订阅",
        "记得订阅",
        "别忘了订阅",
        "点击订阅",
        "喜欢就订阅",
        "感谢您的观看",
        "感谢您的收看",
        "下期再见",
        "我们下期再见",
        "下次再见",
        "再见",  # Only when standalone
        "字幕由",
        "字幕制作",
        "字幕提供",
        "机翻字幕",
        "自动字幕",
    ]

    # Common Whisper hallucinations in English
    ENGLISH_HALLUCINATIONS = [
        "thanks for watching",
        "thank you for watching",
        "subscribe to",
        "please subscribe",
        "like and subscribe",
        "don't forget to subscribe",
        "hit the subscribe button",
        "click subscribe",
        "see you next time",
        "see you in the next",
        "subtitles by",
        "captions by",
        "auto-generated",
        "machine translated",
    ]

    # Japanese-specific hallucinations
    JAPANESE_HALLUCINATIONS = [
        "ご視聴ありがとう",
        "ご視聴ありがとうございました",
        "チャンネル登録",
        "チャンネル登録お願いします",
        "高評価",
        "高評価お願いします",
        "チャンネル登録よろしく",
        "ご視聴いただき",
        "ありがとうございました",  # Only when standalone at end
        "次回もお楽しみに",
        "また次回",
        "字幕",
    ]

    # Bracketed annotation patterns (regex)
    BRACKET_PATTERNS = [
        r"^\s*[\[【\(（]\s*音[乐樂]?\s*[\]】\)）]\s*$",  # [音乐], [音樂], (音乐)
        r"^\s*[\[【\(（]\s*笑[声聲]?\s*[\]】\)）]\s*$",  # [笑声], [笑聲]
        r"^\s*[\[【\(（]\s*掌声\s*[\]】\)）]\s*$",  # [掌声]
        r"^\s*[\[【\(（]\s*鼓掌\s*[\]】\)）]\s*$",  # [鼓掌]
        r"^\s*[\[【\(（]\s*沉默\s*[\]】\)）]\s*$",  # [沉默]
        r"^\s*[\[【\(（]\s*静音\s*[\]】\)）]\s*$",  # [静音]
        r"^\s*[\[【\(（]\s*无声\s*[\]】\)）]\s*$",  # [无声]
        r"^\s*[\[【\(（]\s*背景音\s*[\]】\)）]\s*$",  # [背景音]
        r"^\s*[\[【\(（]\s*BGM\s*[\]】\)）]\s*$",  # [BGM]
        r"^\s*\[\s*Music\s*\]\s*$",  # [Music]
        r"^\s*\[\s*Applause\s*\]\s*$",  # [Applause]
        r"^\s*\[\s*Laughter\s*\]\s*$",  # [Laughter]
        r"^\s*\[\s*Silence\s*\]\s*$",  # [Silence]
        r"^\s*\[\s*Inaudible\s*\]\s*$",  # [Inaudible]
        r"^\s*\(\s*Music\s*\)\s*$",  # (Music)
        r"^\s*\(\s*Applause\s*\)\s*$",  # (Applause)
        r"^\s*\(\s*Laughter\s*\)\s*$",  # (Laughter)
        r"^\s*♪+\s*$",  # Musical notes only
        r"^\s*🎵+\s*$",  # Musical emoji only
        r"^\s*🎶+\s*$",  # Musical emoji only
    ]

    # Single meaningless characters (interjections)
    # Only filter these if they appear alone AND have very short duration
    SINGLE_CHAR_INTERJECTIONS = {
        "啊",
        "呃",
        "嗯",
        "哦",
        "噢",
        "唔",
        "嘛",
        "呀",
        "哎",
        "诶",
        "欸",
        "喂",
        "嘿",
        "哈",
        "呵",
        "嗨",
        "哼",
        "咦",
        "哇",
        "呜",
        "嗷",
        "噫",
        "唉",
        "咳",
        "嘶",
        "吧",
        "呢",
        "吗",
        "么",
        "了",
    }

    def __init__(
        self,
        repetition_threshold: int = 5,
        single_char_max_duration: float = 0.5,
        custom_hallucinations: Optional[list[str]] = None,
    ):
        """
        Initialize the hallucination detector.

        Args:
            repetition_threshold: Minimum consecutive same characters to flag
            single_char_max_duration: Max duration for single-char filtering
            custom_hallucinations: Additional hallucination patterns to detect
        """
        self.repetition_threshold = repetition_threshold
        self.single_char_max_duration = single_char_max_duration

        # Compile bracket patterns for efficiency
        self._bracket_regexes = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.BRACKET_PATTERNS
        ]

        # Build combined hallucination set
        self._hallucinations_zh = set(self.CHINESE_HALLUCINATIONS)
        self._hallucinations_en = {h.lower() for h in self.ENGLISH_HALLUCINATIONS}
        self._hallucinations_ja = set(self.JAPANESE_HALLUCINATIONS)

        if custom_hallucinations:
            for h in custom_hallucinations:
                # Add to appropriate set based on character detection
                if any("\u4e00" <= c <= "\u9fff" for c in h):
                    self._hallucinations_zh.add(h)
                elif any("\u3040" <= c <= "\u30ff" for c in h):
                    self._hallucinations_ja.add(h)
                else:
                    self._hallucinations_en.add(h.lower())

        # Compile repetition pattern
        # Matches 5+ consecutive identical characters
        self._repetition_pattern = re.compile(r"(.)\1{" + str(repetition_threshold - 1) + r",}")

    def detect(self, entry: SubtitleEntry) -> DetectionResult:
        """
        Detect if a subtitle entry is a hallucination.

        Applies multiple detection rules in order of specificity.
        Returns the first match with highest confidence.

        Args:
            entry: The subtitle entry to check

        Returns:
            DetectionResult with detection details
        """
        text = entry.text.strip()

        # Check empty/whitespace first
        if not text:
            return DetectionResult(
                is_hallucination=True,
                hallucination_type=HallucinationType.EMPTY_OR_WHITESPACE,
                confidence=1.0,
                reason="Empty or whitespace-only content",
                matched_pattern="",
            )

        # Check bracketed annotations (highest confidence)
        result = self._check_bracketed_annotation(text)
        if result.is_hallucination:
            return result

        # Check repetitive characters
        result = self._check_repetitive_chars(text)
        if result.is_hallucination:
            return result

        # Check common hallucinations (Chinese)
        result = self._check_chinese_hallucinations(text)
        if result.is_hallucination:
            return result

        # Check common hallucinations (English)
        result = self._check_english_hallucinations(text)
        if result.is_hallucination:
            return result

        # Check Japanese hallucinations
        result = self._check_japanese_hallucinations(text)
        if result.is_hallucination:
            return result

        # Check single meaningless characters (most conservative)
        result = self._check_single_meaningless(text, entry.duration)
        if result.is_hallucination:
            return result

        # No hallucination detected
        return DetectionResult(
            is_hallucination=False,
            confidence=0.0,
            reason="No hallucination patterns detected",
        )

    def _check_bracketed_annotation(self, text: str) -> DetectionResult:
        """Check for bracketed annotations like [音乐], (Music), etc."""
        for regex in self._bracket_regexes:
            if regex.match(text):
                return DetectionResult(
                    is_hallucination=True,
                    hallucination_type=HallucinationType.BRACKETED_ANNOTATION,
                    confidence=0.95,
                    reason="Bracketed annotation (non-speech content marker)",
                    matched_pattern=text,
                )
        return DetectionResult(is_hallucination=False)

    def _check_repetitive_chars(self, text: str) -> DetectionResult:
        """Check for repetitive characters (e.g., '谢谢谢谢谢', 'wwwww')."""
        match = self._repetition_pattern.search(text)
        if match:
            repeated_char = match.group(1)
            repeat_count = len(match.group(0))

            # Higher confidence for more repetitions
            confidence = min(0.7 + (repeat_count - self.repetition_threshold) * 0.05, 0.95)

            # Check if the entire text is just repetition
            if text == match.group(0):
                confidence = 0.98

            return DetectionResult(
                is_hallucination=True,
                hallucination_type=HallucinationType.REPETITIVE_CHARS,
                confidence=confidence,
                reason=f"Repetitive character '{repeated_char}' repeated {repeat_count} times",
                matched_pattern=match.group(0),
            )
        return DetectionResult(is_hallucination=False)

    def _check_chinese_hallucinations(self, text: str) -> DetectionResult:
        """Check for common Chinese Whisper hallucinations."""
        text_normalized = text.strip()

        # Exact match (highest confidence)
        if text_normalized in self._hallucinations_zh:
            return DetectionResult(
                is_hallucination=True,
                hallucination_type=HallucinationType.COMMON_HALLUCINATION,
                confidence=0.95,
                reason="Common Whisper hallucination (Chinese)",
                matched_pattern=text_normalized,
            )

        # Check if text contains hallucination as significant portion
        for hallucination in self._hallucinations_zh:
            if hallucination in text_normalized:
                # Calculate what portion of the text is the hallucination
                ratio = len(hallucination) / len(text_normalized)
                if ratio >= 0.7:  # Hallucination is 70%+ of the text
                    return DetectionResult(
                        is_hallucination=True,
                        hallucination_type=HallucinationType.COMMON_HALLUCINATION,
                        confidence=0.85 * ratio,
                        reason=f"Contains common hallucination: '{hallucination}'",
                        matched_pattern=hallucination,
                    )

        return DetectionResult(is_hallucination=False)

    def _check_english_hallucinations(self, text: str) -> DetectionResult:
        """Check for common English Whisper hallucinations."""
        text_lower = text.strip().lower()

        # Exact match
        if text_lower in self._hallucinations_en:
            return DetectionResult(
                is_hallucination=True,
                hallucination_type=HallucinationType.COMMON_HALLUCINATION,
                confidence=0.95,
                reason="Common Whisper hallucination (English)",
                matched_pattern=text,
            )

        # Check if text contains hallucination as significant portion
        for hallucination in self._hallucinations_en:
            if hallucination in text_lower:
                ratio = len(hallucination) / len(text_lower)
                if ratio >= 0.7:
                    return DetectionResult(
                        is_hallucination=True,
                        hallucination_type=HallucinationType.COMMON_HALLUCINATION,
                        confidence=0.85 * ratio,
                        reason=f"Contains common hallucination: '{hallucination}'",
                        matched_pattern=hallucination,
                    )

        return DetectionResult(is_hallucination=False)

    def _check_japanese_hallucinations(self, text: str) -> DetectionResult:
        """Check for common Japanese Whisper hallucinations."""
        text_normalized = text.strip()

        # Exact match
        if text_normalized in self._hallucinations_ja:
            return DetectionResult(
                is_hallucination=True,
                hallucination_type=HallucinationType.JAPANESE_HALLUCINATION,
                confidence=0.95,
                reason="Common Whisper hallucination (Japanese)",
                matched_pattern=text_normalized,
            )

        # Check if text contains hallucination as significant portion
        for hallucination in self._hallucinations_ja:
            if hallucination in text_normalized:
                ratio = len(hallucination) / len(text_normalized)
                if ratio >= 0.7:
                    return DetectionResult(
                        is_hallucination=True,
                        hallucination_type=HallucinationType.JAPANESE_HALLUCINATION,
                        confidence=0.85 * ratio,
                        reason=f"Contains common hallucination: '{hallucination}'",
                        matched_pattern=hallucination,
                    )

        return DetectionResult(is_hallucination=False)

    def _check_single_meaningless(self, text: str, duration: float) -> DetectionResult:
        """
        Check for single meaningless characters.

        Very conservative: only flags single interjection characters
        that have very short duration (likely noise/artifacts).
        """
        # Only check single characters
        if len(text) != 1:
            return DetectionResult(is_hallucination=False)

        # Must be a known interjection
        if text not in self.SINGLE_CHAR_INTERJECTIONS:
            return DetectionResult(is_hallucination=False)

        # Must have very short duration
        if duration > self.single_char_max_duration:
            return DetectionResult(is_hallucination=False)

        # Calculate confidence based on duration
        # Shorter duration = higher confidence it's noise
        confidence = 0.8 + (1 - duration / self.single_char_max_duration) * 0.15

        return DetectionResult(
            is_hallucination=True,
            hallucination_type=HallucinationType.SINGLE_MEANINGLESS,
            confidence=confidence,
            reason=f"Single interjection '{text}' with very short duration ({duration:.2f}s)",
            matched_pattern=text,
        )


class PreFilter:
    """
    Pre-filters subtitle entries using hallucination detection.

    Applies conservative filtering rules to remove obvious hallucinations
    before sending to LLM processing, reducing token usage and improving
    output quality.
    """

    def __init__(
        self,
        detector: Optional[HallucinationDetector] = None,
        min_confidence: float = 0.8,
    ):
        """
        Initialize the pre-filter.

        Args:
            detector: HallucinationDetector instance (creates default if None)
            min_confidence: Minimum confidence threshold for filtering (0.0-1.0)
        """
        self.detector = detector or HallucinationDetector()
        self.min_confidence = min_confidence

    def filter(self, entries: list[SubtitleEntry]) -> PreFilterResult:
        """
        Filter a list of subtitle entries.

        Args:
            entries: List of subtitle entries to filter

        Returns:
            PreFilterResult with kept entries, filtered entries, and stats
        """
        result = PreFilterResult()
        result.stats.total_entries = len(entries)

        for entry in entries:
            detection = self.detector.detect(entry)

            if detection.is_hallucination and detection.confidence >= self.min_confidence:
                # Filter this entry
                result.filtered_entries.append(FilteredEntry(entry=entry, detection=detection))
                result.stats.add_filtered(detection.hallucination_type)
            else:
                # Keep this entry
                result.kept_entries.append(entry)
                result.stats.add_kept()

        return result

    def filter_to_processed(
        self, entries: list[SubtitleEntry]
    ) -> tuple[list[SubtitleEntry], list[ProcessedEntry]]:
        """
        Filter entries and return both kept entries and ProcessedEntry objects
        for filtered items (marked as DELETE).

        This is useful when you want to include filtered items in the final
        output with proper action tracking.

        Args:
            entries: List of subtitle entries to filter

        Returns:
            Tuple of (kept_entries, deleted_processed_entries)
        """
        filter_result = self.filter(entries)

        # Convert filtered entries to ProcessedEntry with DELETE action
        deleted_entries = []
        for filtered in filter_result.filtered_entries:
            processed = ProcessedEntry(
                original_ids=[filtered.entry.id],
                start=filtered.entry.start,
                end=filtered.entry.end,
                text="",
                action=ProcessingAction.DELETE,
                reason=f"Pre-filter: {filtered.detection.reason} (confidence: {filtered.detection.confidence:.2f})",
                original_text=filtered.entry.text,
            )
            deleted_entries.append(processed)

        return filter_result.kept_entries, deleted_entries

    def get_filter_summary(self, result: PreFilterResult) -> str:
        """
        Generate a human-readable summary of filtering results.

        Args:
            result: PreFilterResult from filter()

        Returns:
            Formatted summary string
        """
        lines = [
            f"Pre-filter Summary:",
            f"  Total entries: {result.stats.total_entries}",
            f"  Kept: {result.stats.kept_count}",
            f"  Filtered: {result.stats.filtered_count} ({result.stats.filter_rate:.1f}%)",
        ]

        if result.stats.by_type:
            lines.append("  By type:")
            for h_type, count in sorted(result.stats.by_type.items(), key=lambda x: -x[1]):
                lines.append(f"    - {h_type.value}: {count}")

        return "\n".join(lines)


def create_default_prefilter() -> PreFilter:
    """
    Create a PreFilter with default conservative settings.

    Returns:
        PreFilter instance with default HallucinationDetector
    """
    detector = HallucinationDetector(
        repetition_threshold=5,
        single_char_max_duration=0.5,
    )
    return PreFilter(detector=detector, min_confidence=0.8)
