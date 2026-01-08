"""Main subtitle processor engine for SubtitleLab."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Set

import pysubs2

from .models import (
    SubtitleEntry,
    ProcessedEntry,
    ProcessingAction,
    BatchResult,
    ProcessingStats,
    ProcessingStatus,
    SemanticContext,
    WindowAdjustmentRequest,
)
from .config import AppConfig, LLMConfig, ProcessingConfig, UserPromptConfig
from .preprocessor import PreFilter, create_default_prefilter
from .chunker import SmartChunker, ChunkerConfig, SlidingWindowManager, Batch
from .llm_client import LLMClient, LLMResponse
from .semantic_analyzer import SemanticAnalyzer

logger = logging.getLogger(__name__)


# =============================================================================
# Processing Constants
# =============================================================================

# Processing timeout safety multiplier
TIMEOUT_SAFETY_MULTIPLIER = 1.5
# Threshold for suspicious KEEP ratio (potential lazy LLM)
LAZY_KEEP_THRESHOLD = 0.95
# Minimum batch size to trigger lazy detection
LAZY_DETECTION_MIN_BATCH = 10
# Merge validation thresholds
MERGE_MAX_TIME_GAP = 1.0  # Max seconds between merged entries
MERGE_MAX_TEXT_LENGTH = 40  # Max characters after merge
MERGE_MAX_ENTRIES = 5  # Max entries to merge at once


# =============================================================================
# SYSTEM_PROMPT V2 - Enhanced with deep understanding capabilities
# =============================================================================

SYSTEM_PROMPT = """<role>资深字幕校对专家，专门处理Whisper语音识别生成的日译中字幕。</role>

<core_principle>宁可保守，不可过度修改。不确定时选择keep。</core_principle>

<decision_framework>
## 同音字判断
| 词 | 判断方法 |
|----|----------|
| 他/她 | 根据上下文人物性别，不确定时保持原样 |
| 的/地/得 | 的+名词，地+动词，得+形容词/程度补语 |
| 在/再 | 在=位置/进行时，再=重复/继续 |
| 做/作 | 做=具体动作，作=抽象/书面 |

## 断句合并条件（必须全部满足）
- 时间间隔 < 1秒
- 语义连续（主谓宾不完整）
- 不跨越说话人切换
- 合并后长度 ≤ 40字

## 删除条件（极其保守）
仅删除以下情况：
- 纯噪音：单独语气词且时长<0.5秒
- 明确重复：完全相同内容连续出现
- 系统幻觉：明显的Whisper幻觉（如"感谢观看"出现在内容中间）

## 说话人切换信号
- 时间间隔 > 1.5秒
- 称呼变化（你↔我）
- 语气明显变化
- 问答模式
</decision_framework>

<rules>
1. **完整性**：必须处理每一个输入ID，不可遗漏
2. **保守性**：不确定时选keep
3. **一致性**：同一人名/术语保持一致
4. **时间轴**：合并时取首条start和末条end
5. **格式**：只输出JSON，无其他内容
</rules>

<actions>
| action | 场景 | text |
|--------|------|------|
| keep | 正确无误 | 原文 |
| correct | 错别字/标点/同音字 | 修正后 |
| merge | 断句错误(间隔<1秒) | 合并后 |
| delete | 重复/噪音/幻觉 | "" |
</actions>

<output_format>
{{
  "results": [
    {{
      "original_ids": [ID列表],
      "action": "keep|correct|merge|delete",
      "text": "处理后文本",
      "reason": "简短理由"
    }}
  ],
  "window_adjustment": null
}}
</output_format>

<examples>
输入: [1] 他说的很好 [2] 我也觉得
输出: {{"results":[{{"original_ids":[1],"action":"correct","text":"他说得很好","reason":"的→得"}},{{"original_ids":[2],"action":"keep","text":"我也觉得","reason":"正确"}}],"window_adjustment":null}}

输入: [3] 今天天气 [4] 真不错啊
输出: {{"results":[{{"original_ids":[3,4],"action":"merge","text":"今天天气真不错啊","reason":"断句合并"}}],"window_adjustment":null}}

输入: [5] 嗯
输出: {{"results":[{{"original_ids":[5],"action":"delete","text":"","reason":"噪音"}}],"window_adjustment":null}}
</examples>

<forbidden>
- 不要遗漏任何输入ID
- 不要输出JSON以外的内容
- 不要过度修改正确内容
- 不要合并跨说话人的内容
</forbidden>

{user_context}"""


@dataclass
class ProcessorCallbacks:
    """Callbacks for processor events."""

    on_progress: Optional[Callable[[float, str], None]] = None
    on_batch_complete: Optional[Callable[[BatchResult], None]] = None
    on_log: Optional[Callable[[str, str], None]] = None
    on_error: Optional[Callable[[Exception, str], None]] = None


@dataclass
class ProcessorState:
    """Internal state of the processor."""

    entries: list[SubtitleEntry] = field(default_factory=list)
    filtered_entries: list[ProcessedEntry] = field(default_factory=list)
    processed_entries: list[ProcessedEntry] = field(default_factory=list)
    batch_results: list[BatchResult] = field(default_factory=list)
    stats: ProcessingStats = field(default_factory=ProcessingStats)
    semantic_context: Optional[SemanticContext] = None
    is_cancelled: bool = False


class SubtitleProcessor:
    """Main subtitle processing engine."""

    def __init__(
        self,
        config: AppConfig,
        callbacks: Optional[ProcessorCallbacks] = None,
    ):
        self.config = config
        self.callbacks = callbacks or ProcessorCallbacks()
        self._state = ProcessorState()
        self._llm_client: Optional[LLMClient] = None
        self._prefilter: Optional[PreFilter] = None
        self._chunker: Optional[SmartChunker] = None
        self._window_manager: Optional[SlidingWindowManager] = None
        self._semantic_analyzer: Optional[SemanticAnalyzer] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _log(self, level: str, message: str) -> None:
        logger.log(getattr(logging, level.upper(), logging.INFO), message)
        if self.callbacks.on_log:
            self.callbacks.on_log(level, message)

    def _emit_progress(self, progress: float, message: str) -> None:
        if self.callbacks.on_progress:
            self.callbacks.on_progress(progress, message)

    def _emit_error(self, error: Exception, context: str) -> None:
        logger.error(f"{context}: {error}", exc_info=True)
        if self.callbacks.on_error:
            self.callbacks.on_error(error, context)

    def load_subtitles(self, file_path: str | Path) -> list[SubtitleEntry]:
        """Load subtitles from file using pysubs2."""
        file_path = Path(file_path)
        self._log("info", f"Loading subtitles from {file_path}")

        try:
            subs = pysubs2.load(str(file_path))
        except Exception as e:
            self._emit_error(e, f"Failed to load subtitle file: {file_path}")
            raise

        entries = []
        for i, line in enumerate(subs, start=1):
            entry = SubtitleEntry(
                id=i,
                start=line.start / 1000.0,
                end=line.end / 1000.0,
                text=line.text.replace("\\N", "\n").replace("\\n", "\n"),
                status=ProcessingStatus.PENDING,
            )
            entries.append(entry)

        self._state.entries = entries
        self._log("info", f"Loaded {len(entries)} subtitle entries")
        return entries

    def _init_components(self) -> None:
        """Initialize processing components."""
        self._llm_client = LLMClient(
            config=self.config.llm,
            concurrency=self.config.processing.concurrency,
        )

        if self.config.processing.enable_pre_filter:
            self._prefilter = create_default_prefilter()

        chunker_config = ChunkerConfig(
            context_window=self.config.llm.context_window,
            max_output_tokens=self.config.llm.max_output_tokens,
            min_batch_size=self.config.processing.min_window_size,
            max_batch_size=self.config.processing.max_window_size,
            gap_threshold=self.config.processing.scene_gap_threshold,
        )
        self._chunker = SmartChunker(chunker_config)

        self._window_manager = SlidingWindowManager(
            window_size=self.config.processing.window_size,
            overlap=self.config.processing.window_overlap,
            min_window_size=self.config.processing.min_window_size,
            max_window_size=self.config.processing.max_window_size,
            allow_dynamic_adjustment=self.config.processing.allow_dynamic_window,
        )

        if self.config.processing.enable_semantic_analysis:
            self._semantic_analyzer = SemanticAnalyzer(self._llm_client)

        self._semaphore = asyncio.Semaphore(self.config.processing.concurrency)

    def _build_system_prompt(self) -> str:
        """Build system prompt with user context."""
        user_context = ""
        if self.config.user_prompt:
            user_section = self.config.user_prompt.to_prompt_section()
            if user_section:
                user_context = f"\n## 用户提供的上下文\n\n{user_section}"

        if self._state.semantic_context:
            semantic_section = self._state.semantic_context.to_prompt_context()
            if semantic_section:
                user_context += f"\n\n## 内容分析\n\n{semantic_section}"

        return SYSTEM_PROMPT.format(user_context=user_context)

    def _build_batch_prompt(self, batch: Batch) -> str:
        """Build user prompt for a batch."""
        lines = ["请处理以下字幕：\n"]

        if batch.previous_context:
            lines.append("【上文参考（已处理，仅供参考）】")
            for entry in batch.previous_context:
                ids_str = ",".join(str(i) for i in entry.original_ids)
                lines.append(f"[{ids_str}] {entry.text}")
            lines.append("")

        lines.append("【待处理字幕】")
        for entry in batch.entries:
            time_str = f"{entry.start:.2f}-{entry.end:.2f}"
            lines.append(f"[{entry.id}] ({time_str}) {entry.text}")

        if batch.is_last_batch:
            lines.append("\n注意：这是最后一批字幕。")

        return "\n".join(lines)

    def _parse_llm_response(
        self,
        response: LLMResponse,
        batch: Batch,
    ) -> tuple[list[ProcessedEntry], Optional[WindowAdjustmentRequest]]:
        """Parse LLM response into processed entries."""
        content = response.content
        if isinstance(content, str):
            import json

            content = json.loads(content)

        results = content.get("results", [])
        entries = []
        entry_map = {e.id: e for e in batch.entries}

        if not isinstance(results, list):
            self._log("warning", f"Invalid results format: expected list, got {type(results)}")
            results = []

        for item in results:
            if isinstance(item, str):
                try:
                    import json

                    item = json.loads(item)
                except json.JSONDecodeError:
                    self._log(
                        "warning", f"Skipping invalid item (not dict or valid json string): {item}"
                    )
                    continue

            if not isinstance(item, dict):
                self._log("warning", f"Skipping invalid item type: {type(item)}")
                continue

            original_ids = item.get("original_ids", [])
            action_str = item.get("action", "keep").lower()
            text = item.get("text", "")
            reason = item.get("reason", "")

            action = ProcessingAction(action_str)

            if action == ProcessingAction.DELETE:
                for oid in original_ids:
                    if oid in entry_map:
                        orig = entry_map[oid]
                        entries.append(
                            ProcessedEntry(
                                original_ids=[oid],
                                start=orig.start,
                                end=orig.end,
                                text="",
                                action=action,
                                reason=reason,
                                original_text=orig.text,
                            )
                        )
            else:
                if not original_ids:
                    continue

                first_id = original_ids[0]
                last_id = original_ids[-1]

                if first_id not in entry_map:
                    continue

                start = entry_map[first_id].start
                end = entry_map[last_id].end if last_id in entry_map else entry_map[first_id].end

                original_texts = [entry_map[oid].text for oid in original_ids if oid in entry_map]
                original_text = (
                    " | ".join(original_texts)
                    if len(original_texts) > 1
                    else (original_texts[0] if original_texts else "")
                )

                entries.append(
                    ProcessedEntry(
                        original_ids=original_ids,
                        start=start,
                        end=end,
                        text=text,
                        action=action,
                        reason=reason,
                        original_text=original_text,
                    )
                )

        window_adj = None
        adj_data = content.get("window_adjustment")
        if adj_data:
            window_adj = WindowAdjustmentRequest(
                action=adj_data.get("action", "expand"),
                amount=adj_data.get("amount", 5),
                reason=adj_data.get("reason", ""),
            )

        return entries, window_adj

    async def _process_batch(self, batch: Batch) -> BatchResult:
        """Process a single batch with timeout protection."""
        start_time = time.time()
        result = BatchResult(batch_index=batch.batch_index)

        # Calculate total timeout: base timeout * (max_retries + 1) * safety multiplier
        total_timeout = (
            self.config.llm.timeout * (self.config.llm.max_retries + 1) * TIMEOUT_SAFETY_MULTIPLIER
        )

        try:
            # Wrap the entire batch processing with timeout
            result = await asyncio.wait_for(
                self._process_batch_internal(batch, result), timeout=total_timeout
            )
        except asyncio.TimeoutError:
            result.success = False
            result.error = f"Batch processing timeout after {total_timeout:.0f}s"
            self._emit_error(TimeoutError(result.error), f"Batch {batch.batch_index} timeout")
        except Exception as e:
            result.success = False
            result.error = str(e)
            self._emit_error(e, f"Batch {batch.batch_index} processing failed")

        result.processing_time = time.time() - start_time
        return result

    async def _process_batch_internal(self, batch: Batch, result: BatchResult) -> BatchResult:
        """Internal batch processing logic."""
        if not self._llm_client:
            raise RuntimeError("LLM client not initialized")

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_batch_prompt(batch)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self._llm_client.chat_completion(
            messages=messages,
            json_mode=True,
            required_fields=["results"],
        )

        entries, window_adj = self._parse_llm_response(response, batch)

        entries = self._ensure_complete_coverage(entries, batch)

        entries = self._validate_merges(entries, batch)

        self._check_lazy_processing(entries, batch)

        if window_adj and self._window_manager:
            if self._window_manager.apply_adjustment(window_adj):
                self._log("info", f"Window adjusted: {window_adj.action} by {window_adj.amount}")

        result.entries = entries
        result.input_tokens = response.input_tokens
        result.output_tokens = response.output_tokens
        result.retries = response.retry_count
        result.success = True

        return result

    def _ensure_complete_coverage(
        self,
        entries: list[ProcessedEntry],
        batch: Batch,
    ) -> list[ProcessedEntry]:
        """
        Ensure all input IDs are covered in the output.

        If LLM missed any IDs, automatically create KEEP entries for them.
        This prevents data loss due to LLM laziness or errors.
        """
        input_ids: Set[int] = {e.id for e in batch.entries}
        processed_ids: Set[int] = set()

        for entry in entries:
            processed_ids.update(entry.original_ids)

        missing_ids = input_ids - processed_ids

        if missing_ids:
            self._log(
                "warning",
                f"LLM missed {len(missing_ids)} IDs: {sorted(missing_ids)}. Auto-completing with KEEP.",
            )

            entry_map = {e.id: e for e in batch.entries}
            for mid in sorted(missing_ids):
                if mid in entry_map:
                    orig = entry_map[mid]
                    entries.append(
                        ProcessedEntry(
                            original_ids=[mid],
                            start=orig.start,
                            end=orig.end,
                            text=orig.text,
                            action=ProcessingAction.KEEP,
                            reason="[Auto] LLM未处理，保持原样",
                            original_text=orig.text,
                        )
                    )

        return entries

    def _check_lazy_processing(
        self,
        entries: list[ProcessedEntry],
        batch: Batch,
    ) -> None:
        """
        Check if LLM is being lazy (returning too many KEEPs).

        Logs a warning if KEEP ratio exceeds threshold on sufficiently large batches.
        """
        if len(batch.entries) < LAZY_DETECTION_MIN_BATCH:
            return

        keep_count = sum(1 for e in entries if e.action == ProcessingAction.KEEP)
        keep_ratio = keep_count / len(entries) if entries else 0

        if keep_ratio > LAZY_KEEP_THRESHOLD:
            self._log(
                "warning",
                f"Batch {batch.batch_index}: KEEP ratio {keep_ratio:.1%} exceeds threshold. "
                f"LLM may not be processing carefully. Consider manual review.",
            )

    def _validate_merges(
        self,
        entries: list[ProcessedEntry],
        batch: Batch,
    ) -> list[ProcessedEntry]:
        entry_map = {e.id: e for e in batch.entries}
        validated = []

        for entry in entries:
            if entry.action != ProcessingAction.MERGE or len(entry.original_ids) <= 1:
                validated.append(entry)
                continue

            if len(entry.original_ids) > MERGE_MAX_ENTRIES:
                self._log(
                    "warning",
                    f"Merge rejected: {len(entry.original_ids)} entries exceeds max {MERGE_MAX_ENTRIES}",
                )
                validated.extend(self._split_invalid_merge(entry, entry_map))
                continue

            if len(entry.text) > MERGE_MAX_TEXT_LENGTH:
                self._log(
                    "warning",
                    f"Merge rejected: {len(entry.text)} chars exceeds max {MERGE_MAX_TEXT_LENGTH}",
                )
                validated.extend(self._split_invalid_merge(entry, entry_map))
                continue

            max_gap = 0.0
            sorted_ids = sorted(entry.original_ids)
            for i in range(1, len(sorted_ids)):
                prev_id, curr_id = sorted_ids[i - 1], sorted_ids[i]
                if prev_id in entry_map and curr_id in entry_map:
                    gap = entry_map[curr_id].start - entry_map[prev_id].end
                    max_gap = max(max_gap, gap)

            if max_gap > MERGE_MAX_TIME_GAP:
                self._log(
                    "warning",
                    f"Merge rejected: {max_gap:.2f}s gap exceeds max {MERGE_MAX_TIME_GAP}s",
                )
                validated.extend(self._split_invalid_merge(entry, entry_map))
                continue

            validated.append(entry)

        return validated

    def _split_invalid_merge(
        self,
        entry: ProcessedEntry,
        entry_map: dict[int, SubtitleEntry],
    ) -> list[ProcessedEntry]:
        result = []
        for oid in entry.original_ids:
            if oid in entry_map:
                orig = entry_map[oid]
                result.append(
                    ProcessedEntry(
                        original_ids=[oid],
                        start=orig.start,
                        end=orig.end,
                        text=orig.text,
                        action=ProcessingAction.KEEP,
                        reason="[Auto] 合并被拒绝，保持原样",
                        original_text=orig.text,
                    )
                )
        return result

    async def _run_semantic_analysis(self, entries: list[SubtitleEntry]) -> None:
        """Run semantic analysis on entries."""
        if not self._semantic_analyzer or not self.config.processing.enable_semantic_analysis:
            return

        self._log("info", "Running semantic analysis...")
        self._emit_progress(0.05, "Analyzing content...")

        try:
            self._state.semantic_context = await self._semantic_analyzer.analyze(entries)
            self._log("info", "Semantic analysis complete")
        except Exception as e:
            self._log("warning", f"Semantic analysis failed: {e}")
            self._state.semantic_context = None

    def _run_prefilter(self, entries: list[SubtitleEntry]) -> list[SubtitleEntry]:
        """Run pre-filter on entries."""
        if not self._prefilter or not self.config.processing.enable_pre_filter:
            return entries

        self._log("info", "Running pre-filter...")
        kept, filtered = self._prefilter.filter_to_processed(entries)
        self._state.filtered_entries = filtered

        self._log("info", f"Pre-filter: kept {len(kept)}, filtered {len(filtered)}")
        return kept

    def _merge_results(self) -> list[ProcessedEntry]:
        """Merge all batch results and filtered entries."""
        all_entries: list[ProcessedEntry] = []

        all_entries.extend(self._state.filtered_entries)

        for batch_result in self._state.batch_results:
            if batch_result.success:
                all_entries.extend(batch_result.entries)

        all_entries.sort(key=lambda e: (e.start, e.original_ids[0] if e.original_ids else 0))
        return all_entries

    def _validate_results(self, entries: list[ProcessedEntry]) -> list[ProcessedEntry]:
        """Validate and fix any timeline overlaps."""
        if not entries:
            return entries

        validated = []
        for i, entry in enumerate(entries):
            if entry.action == ProcessingAction.DELETE:
                validated.append(entry)
                continue

            if validated:
                last_non_deleted = None
                for prev in reversed(validated):
                    if prev.action != ProcessingAction.DELETE:
                        last_non_deleted = prev
                        break

                if last_non_deleted and entry.start < last_non_deleted.end:
                    entry.start = last_non_deleted.end

            if entry.end <= entry.start:
                entry.end = entry.start + 0.1

            validated.append(entry)

        return validated

    async def process(self) -> ProcessingStats:
        """Main async processing pipeline."""
        if not self._state.entries:
            raise ValueError("No subtitles loaded. Call load_subtitles first.")

        self._init_components()
        self._state.stats = ProcessingStats(
            total_entries=len(self._state.entries),
            start_time=datetime.now(),
        )

        self._emit_progress(0.0, "Starting processing...")

        entries = self._run_prefilter(self._state.entries)

        await self._run_semantic_analysis(entries)

        self._log("info", "Creating batches...")
        if not self._chunker:
            raise RuntimeError("Chunker not initialized")

        batches = self._chunker.create_batches(
            entries,
            batch_size=self.config.processing.window_size,
        )
        self._state.stats.total_batches = len(batches)

        self._log("info", f"Processing {len(batches)} batches...")
        self._emit_progress(0.1, f"Processing {len(batches)} batches...")

        async def process_with_semaphore(batch: Batch) -> BatchResult:
            if not self._semaphore:
                raise RuntimeError("Semaphore not initialized")
            async with self._semaphore:
                if self._state.is_cancelled:
                    return BatchResult(
                        batch_index=batch.batch_index, success=False, error="Cancelled"
                    )
                return await self._process_batch(batch)

        # P0-3: Use asyncio.gather to maintain order instead of as_completed
        tasks = [process_with_semaphore(batch) for batch in batches]

        # Process all batches concurrently but collect results in order
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results in order
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Convert exception to failed BatchResult
                batch_result = BatchResult(batch_index=i, success=False, error=str(result))
                self._emit_error(result, f"Batch {i} failed with exception")
            elif isinstance(result, BatchResult):
                batch_result = result
            else:
                # Should not happen
                batch_result = BatchResult(
                    batch_index=i, success=False, error="Unknown result type"
                )

            self._state.batch_results.append(batch_result)
            self._state.stats.update_from_batch(batch_result)

            progress = 0.1 + ((i + 1) / len(batches)) * 0.8
            self._emit_progress(progress, f"Completed batch {i + 1}/{len(batches)}")

            if self.callbacks.on_batch_complete:
                self.callbacks.on_batch_complete(batch_result)

        # P0-3: Sort results by batch_index to ensure correct order
        self._state.batch_results.sort(key=lambda r: r.batch_index)

        self._log("info", "Merging results...")
        self._emit_progress(0.9, "Merging results...")

        merged = self._merge_results()
        validated = self._validate_results(merged)
        self._state.processed_entries = validated

        self._state.stats.processed_entries = len(
            [e for e in validated if e.action != ProcessingAction.DELETE]
        )
        self._state.stats.end_time = datetime.now()

        self._emit_progress(1.0, "Processing complete")
        self._log("info", f"Processing complete: {self._state.stats.processed_entries} entries")

        return self._state.stats

    def save_results(self, output_path: str | Path, format: str = "srt") -> None:
        """Save processed results to file."""
        output_path = Path(output_path)
        self._log("info", f"Saving results to {output_path}")

        subs = pysubs2.SSAFile()

        for entry in self._state.processed_entries:
            if entry.action == ProcessingAction.DELETE:
                continue

            event = pysubs2.SSAEvent(
                start=int(entry.start * 1000),
                end=int(entry.end * 1000),
                text=entry.text.replace("\n", "\\N"),
            )
            subs.append(event)

        subs.save(str(output_path), format_=format)
        self._log("info", f"Saved {len(subs)} entries to {output_path}")

    def cancel(self) -> None:
        """Cancel ongoing processing."""
        self._state.is_cancelled = True
        self._log("info", "Processing cancelled")

    def reset(self) -> None:
        """Reset processor state."""
        self._state = ProcessorState()
        self._log("info", "Processor state reset")

    @property
    def stats(self) -> ProcessingStats:
        return self._state.stats

    @property
    def processed_entries(self) -> list[ProcessedEntry]:
        return self._state.processed_entries

    @property
    def is_processing(self) -> bool:
        return (
            self._state.stats.start_time is not None
            and self._state.stats.end_time is None
            and not self._state.is_cancelled
        )

    async def close(self) -> None:
        """Close resources."""
        if self._llm_client:
            await self._llm_client.close()

    async def __aenter__(self) -> "SubtitleProcessor":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def get_diff_report(self) -> list[dict]:
        result = []
        for entry in self._state.processed_entries:
            if entry.action == ProcessingAction.KEEP:
                continue

            diff_entry = {
                "ids": entry.original_ids,
                "action": entry.action.value,
                "original": entry.original_text or "",
                "processed": entry.text,
                "reason": entry.reason or "",
                "time": f"{entry.start:.2f}-{entry.end:.2f}",
            }
            result.append(diff_entry)
        return result

    def get_diff_summary(self) -> dict:
        total = len(self._state.processed_entries)
        by_action = {
            "keep": 0,
            "correct": 0,
            "merge": 0,
            "delete": 0,
        }

        for entry in self._state.processed_entries:
            action_key = entry.action.value
            if action_key in by_action:
                by_action[action_key] += 1

        return {
            "total_entries": total,
            "unchanged": by_action["keep"],
            "corrected": by_action["correct"],
            "merged": by_action["merge"],
            "deleted": by_action["delete"],
            "change_rate": (total - by_action["keep"]) / total if total > 0 else 0,
        }

    def format_diff_text(self, max_entries: int = 50) -> str:
        lines = []
        diff_entries = self.get_diff_report()[:max_entries]

        for entry in diff_entries:
            action = entry["action"].upper()
            ids = ",".join(str(i) for i in entry["ids"])

            if entry["action"] == "delete":
                lines.append(f'[{ids}] {action}: "{entry["original"]}" → (deleted)')
            elif entry["action"] == "merge":
                lines.append(f'[{ids}] {action}: "{entry["original"]}" → "{entry["processed"]}"')
            else:
                lines.append(f'[{ids}] {action}: "{entry["original"]}" → "{entry["processed"]}"')

            if entry["reason"]:
                lines.append(f"       Reason: {entry['reason']}")

        if len(self.get_diff_report()) > max_entries:
            lines.append(f"... and {len(self.get_diff_report()) - max_entries} more changes")

        return "\n".join(lines)
