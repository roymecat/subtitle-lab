"""
Intelligent chunker module for SubtitleLab.

Handles smart subtitle batching with sliding window support,
semantic boundary detection, and dynamic window adjustment.
"""

from dataclasses import dataclass, field
from typing import Optional

from .models import SubtitleEntry, ProcessedEntry, WindowAdjustmentRequest


# Token estimation constants
CHARS_PER_TOKEN_CHINESE = 1.5  # Chinese characters per token (approximate)
CHARS_PER_TOKEN_ENGLISH = 4.0  # English characters per token (approximate)
DEFAULT_SYSTEM_PROMPT_TOKENS = 2000  # Reserved tokens for system prompt
OUTPUT_BUFFER_RATIO = 1.5  # Output is typically 1.5x input for corrections

# Semantic boundary thresholds
DEFAULT_GAP_THRESHOLD = 1.5  # Seconds - prefer splitting at gaps > this
SCENE_CHANGE_GAP = 3.0  # Seconds - strong indicator of scene change


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens for a given text.

    Uses a heuristic based on character types:
    - Chinese characters: ~1.5 chars per token
    - ASCII/English: ~4 chars per token

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated number of tokens.
    """
    if not text:
        return 0

    chinese_chars = 0
    ascii_chars = 0

    for char in text:
        # Check if character is CJK (Chinese, Japanese, Korean)
        if "\u4e00" <= char <= "\u9fff":
            chinese_chars += 1
        elif "\u3400" <= char <= "\u4dbf":  # CJK Extension A
            chinese_chars += 1
        elif "\uf900" <= char <= "\ufaff":  # CJK Compatibility
            chinese_chars += 1
        else:
            ascii_chars += 1

    chinese_tokens = chinese_chars / CHARS_PER_TOKEN_CHINESE
    ascii_tokens = ascii_chars / CHARS_PER_TOKEN_ENGLISH

    return int(chinese_tokens + ascii_tokens) + 1  # +1 for safety margin


def find_best_split_point(
    entries: list[SubtitleEntry],
    target_index: int,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    search_range: int = 5,
) -> int:
    """
    Find the best split point near the target index.

    Prefers splitting at semantic boundaries (large time gaps between subtitles).

    Args:
        entries: List of subtitle entries.
        target_index: The ideal split index.
        gap_threshold: Minimum gap (seconds) to consider as a good split point.
        search_range: Number of entries to search before/after target.

    Returns:
        The best split index (exclusive - entries[:index] is first batch).
    """
    if not entries or target_index <= 0:
        return target_index

    if target_index >= len(entries):
        return len(entries)

    # Search range around target
    start_search = max(1, target_index - search_range)
    end_search = min(len(entries), target_index + search_range + 1)

    best_index = target_index
    best_gap = 0.0

    # Look for the largest gap in the search range
    for i in range(start_search, end_search):
        if i >= len(entries):
            break

        # Calculate gap between entry i-1 and entry i
        gap = entries[i].start - entries[i - 1].end

        # Prefer gaps above threshold, and larger gaps
        if gap >= gap_threshold and gap > best_gap:
            best_gap = gap
            best_index = i

    # If no good gap found, check for scene change gaps (larger threshold)
    if best_gap < gap_threshold:
        for i in range(start_search, end_search):
            if i >= len(entries):
                break
            gap = entries[i].start - entries[i - 1].end
            if gap >= SCENE_CHANGE_GAP:
                return i

    return best_index


@dataclass
class Batch:
    """
    Represents a batch of subtitle entries for processing.

    Attributes:
        batch_index: Zero-based index of this batch.
        entries: List of SubtitleEntry objects to process.
        previous_context: Processed entries from overlap (for continuity).
        is_last_batch: Whether this is the final batch.
    """

    batch_index: int
    entries: list[SubtitleEntry]
    previous_context: list[ProcessedEntry] = field(default_factory=list)
    is_last_batch: bool = False

    @property
    def entry_count(self) -> int:
        """Number of entries in this batch."""
        return len(self.entries)

    @property
    def first_id(self) -> Optional[int]:
        """ID of the first entry, or None if empty."""
        return self.entries[0].id if self.entries else None

    @property
    def last_id(self) -> Optional[int]:
        """ID of the last entry, or None if empty."""
        return self.entries[-1].id if self.entries else None

    @property
    def total_chars(self) -> int:
        """Total characters in all entries."""
        return sum(e.char_count for e in self.entries)

    @property
    def estimated_tokens(self) -> int:
        """Estimated tokens for all entries."""
        return sum(estimate_tokens(e.text) for e in self.entries)

    @property
    def time_span(self) -> tuple[float, float]:
        """Time span (start, end) of this batch."""
        if not self.entries:
            return (0.0, 0.0)
        return (self.entries[0].start, self.entries[-1].end)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "batch_index": self.batch_index,
            "entries": [e.to_dict() for e in self.entries],
            "previous_context": [e.to_dict() for e in self.previous_context],
            "is_last_batch": self.is_last_batch,
        }


@dataclass
class ChunkerConfig:
    """
    Configuration for the SmartChunker.

    Attributes:
        context_window: Model's total context window in tokens.
        max_output_tokens: Maximum output tokens per request.
        system_prompt_tokens: Reserved tokens for system prompt.
        target_batch_size: Target number of entries per batch (if set).
        min_batch_size: Minimum entries per batch.
        max_batch_size: Maximum entries per batch.
        gap_threshold: Time gap threshold for semantic boundaries.
        avg_chars_per_entry: Average characters per subtitle entry.
    """

    context_window: int = 128000
    max_output_tokens: int = 16384
    system_prompt_tokens: int = DEFAULT_SYSTEM_PROMPT_TOKENS
    target_batch_size: Optional[int] = None
    min_batch_size: int = 10
    max_batch_size: int = 100
    gap_threshold: float = DEFAULT_GAP_THRESHOLD
    avg_chars_per_entry: float = 30.0  # Typical for Chinese subtitles


class SmartChunker:
    """
    Intelligent subtitle chunker that calculates optimal batch sizes.

    Considers model context window, token estimation, and semantic
    boundaries when splitting subtitles into batches.
    """

    def __init__(self, config: Optional[ChunkerConfig] = None):
        """
        Initialize the SmartChunker.

        Args:
            config: Chunker configuration. Uses defaults if not provided.
        """
        self.config = config or ChunkerConfig()

    def calculate_optimal_batch_size(
        self,
        entries: list[SubtitleEntry],
        semantic_context_tokens: int = 0,
    ) -> int:
        """
        Calculate the optimal batch size based on model constraints.

        Args:
            entries: All subtitle entries (used to calculate averages).
            semantic_context_tokens: Additional tokens for semantic context.

        Returns:
            Optimal number of entries per batch.
        """
        if not entries:
            return self.config.min_batch_size

        # Calculate average tokens per entry from actual data
        total_tokens = sum(estimate_tokens(e.text) for e in entries)
        avg_tokens_per_entry = total_tokens / len(entries) if entries else 10

        # Account for JSON structure overhead (~50 tokens per entry)
        json_overhead_per_entry = 50
        tokens_per_entry = avg_tokens_per_entry + json_overhead_per_entry

        # Calculate available tokens for input
        # Reserve space for: system prompt, semantic context, output
        reserved_tokens = (
            self.config.system_prompt_tokens
            + semantic_context_tokens
            + self.config.max_output_tokens
        )

        available_input_tokens = self.config.context_window - reserved_tokens

        # Apply safety margin (use 80% of available)
        safe_input_tokens = int(available_input_tokens * 0.8)

        # Calculate batch size
        if tokens_per_entry > 0:
            optimal_size = int(safe_input_tokens / tokens_per_entry)
        else:
            optimal_size = self.config.max_batch_size

        # Clamp to configured bounds
        optimal_size = max(self.config.min_batch_size, optimal_size)
        optimal_size = min(self.config.max_batch_size, optimal_size)

        # Use target if explicitly set and within bounds
        if self.config.target_batch_size is not None:
            target = self.config.target_batch_size
            if self.config.min_batch_size <= target <= optimal_size:
                return target

        return optimal_size

    def create_batches(
        self,
        entries: list[SubtitleEntry],
        batch_size: Optional[int] = None,
        overlap: int = 0,
    ) -> list[Batch]:
        """
        Create batches from subtitle entries with semantic boundary awareness.

        Args:
            entries: List of all subtitle entries.
            batch_size: Entries per batch. Auto-calculated if None.
            overlap: Number of entries to overlap between batches.

        Returns:
            List of Batch objects.
        """
        if not entries:
            return []

        # Calculate batch size if not provided
        if batch_size is None:
            batch_size = self.calculate_optimal_batch_size(entries)

        batches: list[Batch] = []
        current_index = 0
        batch_index = 0

        while current_index < len(entries):
            # Calculate target end index
            target_end = current_index + batch_size

            # Find best split point (semantic boundary)
            if target_end < len(entries):
                actual_end = find_best_split_point(
                    entries,
                    target_end,
                    gap_threshold=self.config.gap_threshold,
                )
            else:
                actual_end = len(entries)

            # Extract batch entries
            batch_entries = entries[current_index:actual_end]

            # Create batch
            batch = Batch(
                batch_index=batch_index,
                entries=batch_entries,
                previous_context=[],  # Will be filled by SlidingWindowManager
                is_last_batch=(actual_end >= len(entries)),
            )

            batches.append(batch)

            # Move to next batch, accounting for overlap
            if overlap > 0 and actual_end < len(entries):
                current_index = actual_end - overlap
            else:
                current_index = actual_end

            batch_index += 1

        return batches

    def estimate_total_tokens(self, entries: list[SubtitleEntry]) -> int:
        """
        Estimate total tokens for all entries.

        Args:
            entries: List of subtitle entries.

        Returns:
            Estimated total tokens.
        """
        return sum(estimate_tokens(e.text) for e in entries)


class SlidingWindowManager:
    """
    Manages sliding window state across batches.

    Tracks previous context for continuity and supports dynamic
    window adjustment requests from the LLM.
    """

    def __init__(
        self,
        window_size: int = 20,
        overlap: int = 3,
        min_window_size: int = 10,
        max_window_size: int = 50,
        allow_dynamic_adjustment: bool = True,
    ):
        """
        Initialize the SlidingWindowManager.

        Args:
            window_size: Default window size (number of entries).
            overlap: Number of overlapping entries between windows.
            min_window_size: Minimum allowed window size.
            max_window_size: Maximum allowed window size.
            allow_dynamic_adjustment: Allow LLM to request adjustments.
        """
        self.window_size = window_size
        self.overlap = overlap
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.allow_dynamic_adjustment = allow_dynamic_adjustment

        # State tracking
        self._current_index = 0
        self._previous_context: list[ProcessedEntry] = []
        self._processed_count = 0
        self._adjustment_history: list[WindowAdjustmentRequest] = []

    @property
    def current_window_size(self) -> int:
        """Current effective window size."""
        return self.window_size

    @property
    def processed_count(self) -> int:
        """Number of entries processed so far."""
        return self._processed_count

    @property
    def previous_context(self) -> list[ProcessedEntry]:
        """Previous context entries for continuity."""
        return self._previous_context.copy()

    def reset(self) -> None:
        """Reset the window manager state."""
        self._current_index = 0
        self._previous_context = []
        self._processed_count = 0
        self._adjustment_history = []

    def get_next_window(
        self,
        entries: list[SubtitleEntry],
    ) -> Optional[Batch]:
        """
        Get the next window of entries to process.

        Args:
            entries: All subtitle entries.

        Returns:
            Next Batch to process, or None if all entries processed.
        """
        if self._current_index >= len(entries):
            return None

        # Calculate window end
        window_end = min(self._current_index + self.window_size, len(entries))

        # Find semantic boundary for split
        if window_end < len(entries):
            window_end = find_best_split_point(
                entries,
                window_end,
                gap_threshold=DEFAULT_GAP_THRESHOLD,
            )

        # Extract window entries
        window_entries = entries[self._current_index : window_end]

        # Create batch with context
        batch = Batch(
            batch_index=self._processed_count,
            entries=window_entries,
            previous_context=self._previous_context.copy(),
            is_last_batch=(window_end >= len(entries)),
        )

        return batch

    def advance_window(
        self,
        processed_entries: list[ProcessedEntry],
        entries_consumed: int,
    ) -> None:
        """
        Advance the window after processing a batch.

        Args:
            processed_entries: Results from processing the current window.
            entries_consumed: Number of original entries that were processed.
        """
        # Update context with overlap entries from processed results
        if self.overlap > 0 and processed_entries:
            # Keep last N processed entries as context
            self._previous_context = processed_entries[-self.overlap :]
        else:
            self._previous_context = []

        # Advance index, accounting for overlap
        if self.overlap > 0:
            self._current_index += entries_consumed - self.overlap
        else:
            self._current_index += entries_consumed

        self._processed_count += 1

    def validate_adjustment(
        self,
        request: WindowAdjustmentRequest,
    ) -> tuple[bool, str]:
        """
        Validate a window adjustment request.

        Args:
            request: The adjustment request from LLM.

        Returns:
            Tuple of (is_valid, message).
        """
        if not self.allow_dynamic_adjustment:
            return False, "Dynamic window adjustment is disabled"

        if request.action not in ("expand", "shrink"):
            return False, f"Invalid action: {request.action}. Must be 'expand' or 'shrink'"

        if request.amount <= 0:
            return False, f"Invalid amount: {request.amount}. Must be positive"

        # Calculate new size
        if request.action == "expand":
            new_size = self.window_size + request.amount
        else:
            new_size = self.window_size - request.amount

        # Check bounds
        if new_size < self.min_window_size:
            return False, f"Cannot shrink below minimum ({self.min_window_size})"

        if new_size > self.max_window_size:
            return False, f"Cannot expand above maximum ({self.max_window_size})"

        return True, "Adjustment is valid"

    def apply_adjustment(
        self,
        request: WindowAdjustmentRequest,
    ) -> bool:
        """
        Apply a window adjustment request.

        Args:
            request: The adjustment request from LLM.

        Returns:
            True if adjustment was applied, False otherwise.
        """
        is_valid, message = self.validate_adjustment(request)

        if not is_valid:
            return False

        # Apply adjustment
        if request.action == "expand":
            self.window_size += request.amount
        else:
            self.window_size -= request.amount

        # Record in history
        self._adjustment_history.append(request)

        return True

    def get_adjustment_history(self) -> list[WindowAdjustmentRequest]:
        """Get the history of window adjustments."""
        return self._adjustment_history.copy()

    def create_all_batches(
        self,
        entries: list[SubtitleEntry],
    ) -> list[Batch]:
        """
        Create all batches for the given entries.

        This is a convenience method that creates all batches upfront.
        For dynamic processing, use get_next_window() and advance_window().

        Args:
            entries: All subtitle entries.

        Returns:
            List of all batches.
        """
        self.reset()
        batches: list[Batch] = []

        while True:
            batch = self.get_next_window(entries)
            if batch is None:
                break

            batches.append(batch)

            # Simulate advancement (no actual processed entries)
            # In real usage, advance_window would be called with actual results
            self._current_index += len(batch.entries)
            if self.overlap > 0 and not batch.is_last_batch:
                self._current_index -= self.overlap
            self._processed_count += 1

        self.reset()  # Reset for actual processing
        return batches
