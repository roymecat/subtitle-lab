"""Quality validation module for SubtitleLab."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .models import ProcessedEntry, ProcessingAction, SubtitleEntry

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    entry_id: int
    severity: ValidationSeverity
    code: str
    message: str
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    is_valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.error_count += 1
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warning_count += 1


@dataclass
class ValidatorConfig:
    max_text_length: int = 80
    min_duration: float = 0.3
    max_duration: float = 10.0
    max_chars_per_second: float = 25.0
    min_gap_between_entries: float = 0.04
    check_timeline_overlap: bool = True
    check_text_length: bool = True
    check_duration: bool = True
    check_reading_speed: bool = True


class QualityValidator:
    def __init__(self, config: Optional[ValidatorConfig] = None):
        self.config = config or ValidatorConfig()

    def validate_entries(
        self,
        entries: list[ProcessedEntry],
        original_entries: Optional[list[SubtitleEntry]] = None,
    ) -> ValidationResult:
        result = ValidationResult()

        for i, entry in enumerate(entries):
            if entry.action == ProcessingAction.DELETE:
                continue

            self._validate_single_entry(entry, result)

            if i > 0:
                prev_entry = entries[i - 1]
                if prev_entry.action != ProcessingAction.DELETE:
                    self._validate_entry_pair(prev_entry, entry, result)

        if original_entries:
            self._validate_coverage(entries, original_entries, result)

        return result

    def _validate_single_entry(self, entry: ProcessedEntry, result: ValidationResult) -> None:
        entry_id = entry.original_ids[0] if entry.original_ids else 0

        if self.config.check_text_length and len(entry.text) > self.config.max_text_length:
            result.add_issue(
                ValidationIssue(
                    entry_id=entry_id,
                    severity=ValidationSeverity.WARNING,
                    code="TEXT_TOO_LONG",
                    message=f"Text length {len(entry.text)} exceeds max {self.config.max_text_length}",
                    suggestion="Consider splitting into multiple entries",
                )
            )

        duration = entry.end - entry.start

        if self.config.check_duration:
            if duration < self.config.min_duration:
                result.add_issue(
                    ValidationIssue(
                        entry_id=entry_id,
                        severity=ValidationSeverity.WARNING,
                        code="DURATION_TOO_SHORT",
                        message=f"Duration {duration:.2f}s below min {self.config.min_duration}s",
                    )
                )
            elif duration > self.config.max_duration:
                result.add_issue(
                    ValidationIssue(
                        entry_id=entry_id,
                        severity=ValidationSeverity.WARNING,
                        code="DURATION_TOO_LONG",
                        message=f"Duration {duration:.2f}s exceeds max {self.config.max_duration}s",
                    )
                )

        if self.config.check_reading_speed and duration > 0 and entry.text:
            chars_per_second = len(entry.text) / duration
            if chars_per_second > self.config.max_chars_per_second:
                result.add_issue(
                    ValidationIssue(
                        entry_id=entry_id,
                        severity=ValidationSeverity.WARNING,
                        code="READING_SPEED_TOO_FAST",
                        message=f"Reading speed {chars_per_second:.1f} cps exceeds max {self.config.max_chars_per_second}",
                        suggestion="Text may be too long for the duration",
                    )
                )

        if entry.end <= entry.start:
            result.add_issue(
                ValidationIssue(
                    entry_id=entry_id,
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_TIMELINE",
                    message=f"End time {entry.end:.2f} <= start time {entry.start:.2f}",
                )
            )

    def _validate_entry_pair(
        self,
        prev: ProcessedEntry,
        curr: ProcessedEntry,
        result: ValidationResult,
    ) -> None:
        if not self.config.check_timeline_overlap:
            return

        curr_id = curr.original_ids[0] if curr.original_ids else 0

        if curr.start < prev.end:
            overlap = prev.end - curr.start
            result.add_issue(
                ValidationIssue(
                    entry_id=curr_id,
                    severity=ValidationSeverity.ERROR,
                    code="TIMELINE_OVERLAP",
                    message=f"Overlaps with previous entry by {overlap:.2f}s",
                )
            )

        gap = curr.start - prev.end
        if 0 < gap < self.config.min_gap_between_entries:
            result.add_issue(
                ValidationIssue(
                    entry_id=curr_id,
                    severity=ValidationSeverity.INFO,
                    code="GAP_TOO_SMALL",
                    message=f"Gap {gap:.3f}s below recommended {self.config.min_gap_between_entries}s",
                )
            )

    def _validate_coverage(
        self,
        processed: list[ProcessedEntry],
        original: list[SubtitleEntry],
        result: ValidationResult,
    ) -> None:
        original_ids = {e.id for e in original}
        processed_ids: set[int] = set()

        for entry in processed:
            processed_ids.update(entry.original_ids)

        missing = original_ids - processed_ids
        if missing:
            for mid in sorted(missing):
                result.add_issue(
                    ValidationIssue(
                        entry_id=mid,
                        severity=ValidationSeverity.ERROR,
                        code="MISSING_ENTRY",
                        message=f"Original entry {mid} not found in processed results",
                    )
                )

    def get_summary(self, result: ValidationResult) -> str:
        if result.is_valid and result.warning_count == 0:
            return "All entries passed validation"

        parts = []
        if result.error_count > 0:
            parts.append(f"{result.error_count} errors")
        if result.warning_count > 0:
            parts.append(f"{result.warning_count} warnings")

        return f"Validation: {', '.join(parts)}"


def create_quality_validator(strict: bool = False) -> QualityValidator:
    if strict:
        config = ValidatorConfig(
            max_text_length=60,
            max_chars_per_second=20.0,
            min_gap_between_entries=0.08,
        )
    else:
        config = ValidatorConfig()

    return QualityValidator(config)
