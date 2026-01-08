"""Speaker detection module for SubtitleLab."""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

from .models import SubtitleEntry

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class SpeakerChangeSignal(str, Enum):
    TIME_GAP = "time_gap"
    PUNCTUATION = "punctuation"
    STYLE_SHIFT = "style_shift"
    EXPLICIT_MARKER = "explicit_marker"
    LLM_DETECTED = "llm_detected"


@dataclass
class SpeakerInfo:
    id: str
    name: Optional[str] = None
    characteristics: list[str] = field(default_factory=list)
    entry_ids: list[int] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        return self.name or f"Speaker {self.id}"


@dataclass
class SpeakerSegment:
    start_entry_id: int
    end_entry_id: int
    speaker_id: str
    confidence: float = 1.0
    change_signal: Optional[SpeakerChangeSignal] = None


@dataclass
class SpeakerDetectionResult:
    speakers: list[SpeakerInfo] = field(default_factory=list)
    segments: list[SpeakerSegment] = field(default_factory=list)
    entry_speaker_map: dict[int, str] = field(default_factory=dict)

    def get_speaker_for_entry(self, entry_id: int) -> Optional[str]:
        return self.entry_speaker_map.get(entry_id)

    def get_speaker_info(self, speaker_id: str) -> Optional[SpeakerInfo]:
        for speaker in self.speakers:
            if speaker.id == speaker_id:
                return speaker
        return None


@dataclass
class SpeakerDetectorConfig:
    time_gap_threshold: float = 2.0
    min_segment_entries: int = 2
    use_llm_detection: bool = True
    max_speakers: int = 10


SPEAKER_DETECTION_PROMPT = """<task>分析字幕内容，识别不同说话人及其特征。</task>

<input>
{subtitle_sample}
</input>

<output_schema>
{{
  "speakers": [
    {{
      "id": "A",
      "name": "人物名称（如能识别）或null",
      "characteristics": ["说话特点1", "说话特点2"]
    }}
  ],
  "segments": [
    {{
      "start_id": 1,
      "end_id": 5,
      "speaker_id": "A",
      "confidence": 0.9
    }}
  ]
}}
</output_schema>

<rules>
- 基于语言风格、称呼、上下文推断说话人
- 无法确定时使用通用ID（A、B、C...）
- confidence范围0-1，表示判断确信度
- 只输出JSON
</rules>"""


class SpeakerDetector:
    def __init__(
        self,
        llm_client: Optional["LLMClient"] = None,
        config: Optional[SpeakerDetectorConfig] = None,
    ):
        self.llm_client = llm_client
        self.config = config or SpeakerDetectorConfig()

    def detect_by_rules(self, entries: list[SubtitleEntry]) -> SpeakerDetectionResult:
        if not entries:
            return SpeakerDetectionResult()

        segments: list[SpeakerSegment] = []
        current_speaker = "A"
        segment_start = entries[0].id

        for i in range(1, len(entries)):
            prev_entry = entries[i - 1]
            curr_entry = entries[i]
            change_signal = self._detect_speaker_change(prev_entry, curr_entry)

            if change_signal:
                segments.append(
                    SpeakerSegment(
                        start_entry_id=segment_start,
                        end_entry_id=prev_entry.id,
                        speaker_id=current_speaker,
                        confidence=0.6,
                        change_signal=change_signal,
                    )
                )
                current_speaker = self._next_speaker_id(current_speaker)
                segment_start = curr_entry.id

        segments.append(
            SpeakerSegment(
                start_entry_id=segment_start,
                end_entry_id=entries[-1].id,
                speaker_id=current_speaker,
                confidence=0.6,
            )
        )

        segments = self._merge_short_segments(segments)
        speakers = self._extract_speakers_from_segments(segments)
        entry_map = self._build_entry_speaker_map(segments, entries)

        return SpeakerDetectionResult(
            speakers=speakers,
            segments=segments,
            entry_speaker_map=entry_map,
        )

    def _detect_speaker_change(
        self, prev: SubtitleEntry, curr: SubtitleEntry
    ) -> Optional[SpeakerChangeSignal]:
        time_gap = curr.start - prev.end
        if time_gap >= self.config.time_gap_threshold:
            return SpeakerChangeSignal.TIME_GAP

        explicit_patterns = [
            r"^[-—–]",
            r"^[「『]",
            r"^\([^)]+\)\s*[:：]",
            r"^[A-Z][a-z]*\s*[:：]",
            r"^[\u4e00-\u9fff]{1,4}[:：]",
        ]
        for pattern in explicit_patterns:
            if re.match(pattern, curr.text.strip()):
                return SpeakerChangeSignal.EXPLICIT_MARKER

        if prev.text.strip().endswith(("?", "？")):
            if not curr.text.strip().endswith(("?", "？")):
                return SpeakerChangeSignal.PUNCTUATION

        return None

    def _next_speaker_id(self, current: str) -> str:
        if len(current) == 1 and current.isalpha():
            if current == "Z":
                return "AA"
            return chr(ord(current) + 1)
        return current + "'"

    def _merge_short_segments(self, segments: list[SpeakerSegment]) -> list[SpeakerSegment]:
        if len(segments) <= 1:
            return segments

        merged: list[SpeakerSegment] = []
        for segment in segments:
            segment_length = segment.end_entry_id - segment.start_entry_id + 1
            if segment_length < self.config.min_segment_entries and merged:
                merged[-1].end_entry_id = segment.end_entry_id
                merged[-1].confidence = min(merged[-1].confidence, 0.4)
            else:
                merged.append(segment)

        return merged

    def _extract_speakers_from_segments(self, segments: list[SpeakerSegment]) -> list[SpeakerInfo]:
        speaker_ids = set()
        for segment in segments:
            speaker_ids.add(segment.speaker_id)

        return [
            SpeakerInfo(
                id=sid,
                entry_ids=[
                    eid
                    for seg in segments
                    if seg.speaker_id == sid
                    for eid in range(seg.start_entry_id, seg.end_entry_id + 1)
                ],
            )
            for sid in sorted(speaker_ids)
        ]

    def _build_entry_speaker_map(
        self, segments: list[SpeakerSegment], entries: list[SubtitleEntry]
    ) -> dict[int, str]:
        entry_map: dict[int, str] = {}
        for segment in segments:
            for entry in entries:
                if segment.start_entry_id <= entry.id <= segment.end_entry_id:
                    entry_map[entry.id] = segment.speaker_id
        return entry_map

    async def detect_with_llm(self, entries: list[SubtitleEntry]) -> SpeakerDetectionResult:
        if not self.llm_client:
            logger.warning("No LLM client, falling back to rule-based detection")
            return self.detect_by_rules(entries)

        if not entries:
            return SpeakerDetectionResult()

        sample_entries = self._sample_for_detection(entries)
        sample_text = self._format_entries_for_prompt(sample_entries)

        try:
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的字幕分析助手，擅长识别对话中的不同说话人。",
                },
                {
                    "role": "user",
                    "content": SPEAKER_DETECTION_PROMPT.format(subtitle_sample=sample_text),
                },
            ]

            response = await self.llm_client.chat_completion(
                messages=messages,
                json_mode=True,
            )

            if isinstance(response.content, dict):
                return self._parse_llm_response(response.content, entries)

            logger.warning("Invalid LLM response format, falling back to rules")
            return self.detect_by_rules(entries)

        except Exception as e:
            logger.warning(f"LLM detection failed: {e}, falling back to rules")
            return self.detect_by_rules(entries)

    def _sample_for_detection(
        self, entries: list[SubtitleEntry], max_samples: int = 50
    ) -> list[SubtitleEntry]:
        if len(entries) <= max_samples:
            return entries

        step = len(entries) / max_samples
        return [entries[int(i * step)] for i in range(max_samples)]

    def _format_entries_for_prompt(self, entries: list[SubtitleEntry]) -> str:
        lines = []
        for entry in entries:
            text = entry.text.replace("\n", " ")
            lines.append(f"[{entry.id}] ({entry.start:.1f}s) {text}")
        return "\n".join(lines)

    def _parse_llm_response(
        self, data: dict, entries: list[SubtitleEntry]
    ) -> SpeakerDetectionResult:
        speakers: list[SpeakerInfo] = []
        for sp_data in data.get("speakers", []):
            speakers.append(
                SpeakerInfo(
                    id=sp_data.get("id", "?"),
                    name=sp_data.get("name"),
                    characteristics=sp_data.get("characteristics", []),
                )
            )

        segments: list[SpeakerSegment] = []
        for seg_data in data.get("segments", []):
            segments.append(
                SpeakerSegment(
                    start_entry_id=seg_data.get("start_id", 1),
                    end_entry_id=seg_data.get("end_id", 1),
                    speaker_id=seg_data.get("speaker_id", "A"),
                    confidence=seg_data.get("confidence", 0.8),
                    change_signal=SpeakerChangeSignal.LLM_DETECTED,
                )
            )

        entry_map = self._build_entry_speaker_map(segments, entries)

        for speaker in speakers:
            speaker.entry_ids = [eid for eid, sid in entry_map.items() if sid == speaker.id]

        return SpeakerDetectionResult(
            speakers=speakers,
            segments=segments,
            entry_speaker_map=entry_map,
        )

    async def detect(self, entries: list[SubtitleEntry]) -> SpeakerDetectionResult:
        if self.config.use_llm_detection and self.llm_client:
            return await self.detect_with_llm(entries)
        return self.detect_by_rules(entries)


def create_speaker_detector(
    llm_client: Optional["LLMClient"] = None,
    use_llm: bool = True,
) -> SpeakerDetector:
    config = SpeakerDetectorConfig(use_llm_detection=use_llm and llm_client is not None)
    return SpeakerDetector(llm_client=llm_client, config=config)
