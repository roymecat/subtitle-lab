"""
Core data models for SubtitleLab.

Defines all data structures used throughout the application.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime


class ProcessingAction(str, Enum):
    """Actions that can be performed on subtitle entries."""

    KEEP = "keep"  # Keep original, no changes
    CORRECT = "correct"  # Text correction (typos, punctuation)
    MERGE = "merge"  # Merge multiple entries into one
    SPLIT = "split"  # Split one entry into multiple
    DELETE = "delete"  # Remove the entry entirely


class ProcessingStatus(str, Enum):
    """Status of subtitle processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SubtitleEntry:
    """
    Represents a single subtitle entry.

    Attributes:
        id: Unique identifier (1-based index from SRT)
        start: Start time in seconds
        end: End time in seconds
        text: Subtitle text content
        status: Processing status
    """

    id: int
    start: float
    end: float
    text: str
    status: ProcessingStatus = ProcessingStatus.PENDING

    @property
    def duration(self) -> float:
        """Duration of the subtitle in seconds."""
        return self.end - self.start

    @property
    def char_count(self) -> int:
        """Number of characters in the text."""
        return len(self.text)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SubtitleEntry":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            start=data["start"],
            end=data["end"],
            text=data["text"],
        )


@dataclass
class ProcessedEntry:
    """
    Represents a processed subtitle entry with action information.

    Attributes:
        original_ids: List of original subtitle IDs that were processed
        start: New start time in seconds
        end: New end time in seconds
        text: Processed text content
        action: The action that was performed
        reason: Explanation for the action
        original_text: Original text before processing (for comparison)
    """

    original_ids: list[int]
    start: float
    end: float
    text: str
    action: ProcessingAction
    reason: Optional[str] = None
    original_text: Optional[str] = None

    @property
    def duration(self) -> float:
        """Duration of the subtitle in seconds."""
        return self.end - self.start

    @property
    def is_modified(self) -> bool:
        """Check if the entry was modified."""
        return self.action != ProcessingAction.KEEP

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_ids": self.original_ids,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "action": self.action.value,
            "reason": self.reason,
            "original_text": self.original_text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessedEntry":
        """Create from dictionary."""
        return cls(
            original_ids=data["original_ids"],
            start=data["start"],
            end=data["end"],
            text=data["text"],
            action=ProcessingAction(data["action"]),
            reason=data.get("reason"),
            original_text=data.get("original_text"),
        )


@dataclass
class BatchResult:
    """
    Result of processing a single batch.

    Attributes:
        batch_index: Index of the batch (0-based)
        entries: List of processed entries
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        processing_time: Time taken to process in seconds
        success: Whether the batch was processed successfully
        error: Error message if failed
        retries: Number of retries needed
    """

    batch_index: int
    entries: list[ProcessedEntry] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    processing_time: float = 0.0
    success: bool = True
    error: Optional[str] = None
    retries: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    @property
    def merged_count(self) -> int:
        """Number of merge operations."""
        return sum(1 for e in self.entries if e.action == ProcessingAction.MERGE)

    @property
    def deleted_count(self) -> int:
        """Number of delete operations."""
        return sum(1 for e in self.entries if e.action == ProcessingAction.DELETE)

    @property
    def corrected_count(self) -> int:
        """Number of correction operations."""
        return sum(1 for e in self.entries if e.action == ProcessingAction.CORRECT)


@dataclass
class ProcessingStats:
    """
    Overall statistics for a processing session.

    Attributes:
        total_entries: Total number of original subtitle entries
        processed_entries: Number of entries after processing
        merged_count: Number of merge operations
        deleted_count: Number of delete operations
        corrected_count: Number of corrections made
        total_batches: Total number of batches
        completed_batches: Number of completed batches
        failed_batches: Number of failed batches
        total_input_tokens: Total input tokens used
        total_output_tokens: Total output tokens generated
        total_retries: Total number of retries across all batches
        start_time: Processing start time
        end_time: Processing end time
        estimated_cost: Estimated cost in USD
    """

    total_entries: int = 0
    processed_entries: int = 0
    merged_count: int = 0
    deleted_count: int = 0
    corrected_count: int = 0
    total_batches: int = 0
    completed_batches: int = 0
    failed_batches: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_retries: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_cost: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def progress_percent(self) -> float:
        """Processing progress as percentage."""
        if self.total_batches == 0:
            return 0.0
        return (self.completed_batches / self.total_batches) * 100

    @property
    def elapsed_time(self) -> float:
        """Elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def estimated_remaining_time(self) -> float:
        """Estimated remaining time in seconds."""
        if self.completed_batches == 0:
            return 0.0
        avg_time_per_batch = self.elapsed_time / self.completed_batches
        remaining_batches = self.total_batches - self.completed_batches
        return avg_time_per_batch * remaining_batches

    def update_from_batch(self, batch_result: BatchResult) -> None:
        """Update stats from a batch result."""
        if batch_result.success:
            self.completed_batches += 1
            self.total_input_tokens += batch_result.input_tokens
            self.total_output_tokens += batch_result.output_tokens
            self.merged_count += batch_result.merged_count
            self.deleted_count += batch_result.deleted_count
            self.corrected_count += batch_result.corrected_count
            self.total_retries += batch_result.retries
        else:
            self.failed_batches += 1


@dataclass
class SemanticContext:
    """
    Semantic context extracted from the full subtitle content.

    Generated by LLM pre-analysis to understand the overall content.

    Attributes:
        summary: Brief summary of the content
        characters: List of identified character names
        topics: Main topics/themes discussed
        style: Detected speaking style (formal, casual, etc.)
        language_notes: Notes about language usage
    """

    summary: str = ""
    characters: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    style: str = ""
    language_notes: str = ""

    def to_prompt_context(self) -> str:
        """Convert to a string suitable for inclusion in prompts."""
        parts = []
        if self.summary:
            parts.append(f"内容摘要: {self.summary}")
        if self.characters:
            parts.append(f"出场人物: {', '.join(self.characters)}")
        if self.topics:
            parts.append(f"主要话题: {', '.join(self.topics)}")
        if self.style:
            parts.append(f"说话风格: {self.style}")
        if self.language_notes:
            parts.append(f"语言特点: {self.language_notes}")
        return "\n".join(parts)


@dataclass
class WindowAdjustmentRequest:
    """
    Request from LLM to adjust the sliding window size.

    Attributes:
        action: "expand" or "shrink"
        amount: Number of lines to add or remove
        reason: Explanation for the adjustment
    """

    action: str  # "expand" or "shrink"
    amount: int
    reason: str
