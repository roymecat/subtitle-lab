"""LLM-based quality scoring module for SubtitleLab."""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .models import ProcessedEntry, ProcessingAction

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    entry_id: int
    score: float
    dimensions: dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    needs_revision: bool = False


@dataclass
class ScoringResult:
    entries: list[ScoreResult] = field(default_factory=list)
    average_score: float = 0.0
    revision_needed_count: int = 0

    def add_result(self, result: ScoreResult) -> None:
        self.entries.append(result)
        if result.needs_revision:
            self.revision_needed_count += 1
        total = sum(e.score for e in self.entries)
        self.average_score = total / len(self.entries) if self.entries else 0.0


@dataclass
class ScorerConfig:
    enabled: bool = False
    min_acceptable_score: float = 0.7
    max_retries: int = 2
    score_dimensions: list[str] = field(
        default_factory=lambda: ["accuracy", "fluency", "consistency"]
    )


SCORING_PROMPT = """<task>评估字幕校对结果的质量。</task>

<original>
{original_text}
</original>

<processed>
{processed_text}
</processed>

<action>{action}</action>
<reason>{reason}</reason>

<scoring_dimensions>
- accuracy: 修正是否准确（同音字、标点、语法）
- fluency: 修正后是否通顺自然
- consistency: 是否与上下文风格一致
</scoring_dimensions>

<output_schema>
{{
  "score": 0.0-1.0,
  "dimensions": {{
    "accuracy": 0.0-1.0,
    "fluency": 0.0-1.0,
    "consistency": 0.0-1.0
  }},
  "feedback": "简短评价",
  "needs_revision": true/false
}}
</output_schema>

<rules>
- keep操作默认1.0分
- delete操作需验证是否确实应删除
- 只输出JSON
</rules>"""


class QualityScorer:
    def __init__(
        self,
        llm_client: Optional["LLMClient"] = None,
        config: Optional[ScorerConfig] = None,
    ):
        self.llm_client = llm_client
        self.config = config or ScorerConfig()

    @property
    def is_enabled(self) -> bool:
        return self.config.enabled and self.llm_client is not None

    async def score_entry(self, entry: ProcessedEntry) -> ScoreResult:
        entry_id = entry.original_ids[0] if entry.original_ids else 0

        if entry.action == ProcessingAction.KEEP:
            return ScoreResult(
                entry_id=entry_id,
                score=1.0,
                dimensions={"accuracy": 1.0, "fluency": 1.0, "consistency": 1.0},
                feedback="保持原样，无需评分",
                needs_revision=False,
            )

        if not self.llm_client:
            return ScoreResult(
                entry_id=entry_id,
                score=0.8,
                feedback="无LLM客户端，跳过评分",
                needs_revision=False,
            )

        try:
            prompt = SCORING_PROMPT.format(
                original_text=entry.original_text or "",
                processed_text=entry.text,
                action=entry.action.value,
                reason=entry.reason or "",
            )

            messages = [
                {"role": "system", "content": "你是字幕校对质量评估专家。"},
                {"role": "user", "content": prompt},
            ]

            response = await self.llm_client.chat_completion(
                messages=messages,
                json_mode=True,
            )

            if isinstance(response.content, dict):
                return self._parse_score_response(response.content, entry_id)

            return ScoreResult(
                entry_id=entry_id,
                score=0.8,
                feedback="响应格式错误",
                needs_revision=False,
            )

        except Exception as e:
            logger.warning(f"Scoring failed for entry {entry_id}: {e}")
            return ScoreResult(
                entry_id=entry_id,
                score=0.8,
                feedback=f"评分失败: {str(e)}",
                needs_revision=False,
            )

    def _parse_score_response(self, data: dict, entry_id: int) -> ScoreResult:
        score = float(data.get("score", 0.8))
        dimensions = data.get("dimensions", {})
        feedback = data.get("feedback", "")
        needs_revision = data.get("needs_revision", False)

        if score < self.config.min_acceptable_score:
            needs_revision = True

        return ScoreResult(
            entry_id=entry_id,
            score=score,
            dimensions=dimensions,
            feedback=feedback,
            needs_revision=needs_revision,
        )

    async def score_entries(
        self,
        entries: list[ProcessedEntry],
        skip_keep: bool = True,
    ) -> ScoringResult:
        result = ScoringResult()

        for entry in entries:
            if skip_keep and entry.action == ProcessingAction.KEEP:
                continue

            score_result = await self.score_entry(entry)
            result.add_result(score_result)

        return result

    def get_entries_needing_revision(self, scoring_result: ScoringResult) -> list[int]:
        return [e.entry_id for e in scoring_result.entries if e.needs_revision]


def create_quality_scorer(
    llm_client: Optional["LLMClient"] = None,
    enabled: bool = False,
    min_score: float = 0.7,
) -> QualityScorer:
    config = ScorerConfig(
        enabled=enabled,
        min_acceptable_score=min_score,
    )
    return QualityScorer(llm_client=llm_client, config=config)
