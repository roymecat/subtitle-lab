"""Logger utility for SubtitleLab with GUI integration support."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional, Awaitable
import asyncio
import json


class LogLevel(Enum):
    """Log severity levels."""

    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    SUCCESS = auto()


@dataclass
class LogEntry:
    """A single log entry."""

    timestamp: datetime
    level: LogLevel
    message: str
    details: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert log entry to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.name,
            "message": self.message,
            "details": self.details,
        }

    def format(self, include_details: bool = False) -> str:
        """Format log entry as string."""
        timestamp_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        base = f"[{timestamp_str}] [{self.level.name}] {self.message}"
        if include_details and self.details:
            base += f" | {json.dumps(self.details)}"
        return base


class Logger:
    """
    Logger with in-memory storage for GUI display.

    Supports async callbacks for real-time updates and filtering by level.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        """
        Initialize the logger.

        Args:
            max_entries: Maximum number of log entries to keep in memory.
        """
        self._entries: list[LogEntry] = []
        self._max_entries = max_entries
        self._callbacks: list[Callable[[LogEntry], Awaitable[None] | None]] = []
        self._min_level: LogLevel = LogLevel.DEBUG

    @property
    def entries(self) -> list[LogEntry]:
        """Get all log entries."""
        return self._entries.copy()

    @property
    def min_level(self) -> LogLevel:
        """Get the minimum log level."""
        return self._min_level

    @min_level.setter
    def min_level(self, level: LogLevel) -> None:
        """Set the minimum log level for filtering."""
        self._min_level = level

    def add_callback(self, callback: Callable[[LogEntry], Awaitable[None] | None]) -> None:
        """
        Add a callback for real-time log updates.

        Args:
            callback: Function called with each new log entry.
                     Can be sync or async.
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[LogEntry], Awaitable[None] | None]) -> None:
        """
        Remove a previously added callback.

        Args:
            callback: The callback to remove.
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _log(self, level: LogLevel, message: str, details: Optional[dict] = None) -> LogEntry:
        """
        Internal method to create and store a log entry.

        Args:
            level: The log level.
            message: The log message.
            details: Optional additional details.

        Returns:
            The created LogEntry.
        """
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            details=details,
        )

        # Add entry and enforce max limit
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]

        # Notify callbacks
        self._notify_callbacks(entry)

        return entry

    def _notify_callbacks(self, entry: LogEntry) -> None:
        """Notify all registered callbacks of a new entry."""
        for callback in self._callbacks:
            try:
                result = callback(entry)
                # Handle async callbacks
                if asyncio.iscoroutine(result):
                    # Schedule the coroutine if there's a running event loop
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)
                    except RuntimeError:
                        # No running event loop, run synchronously
                        asyncio.run(result)
            except Exception:
                # Don't let callback errors break logging
                pass

    def debug(self, message: str, details: Optional[dict] = None) -> LogEntry:
        """
        Log a debug message.

        Args:
            message: The log message.
            details: Optional additional details.

        Returns:
            The created LogEntry.
        """
        return self._log(LogLevel.DEBUG, message, details)

    def info(self, message: str, details: Optional[dict] = None) -> LogEntry:
        """
        Log an info message.

        Args:
            message: The log message.
            details: Optional additional details.

        Returns:
            The created LogEntry.
        """
        return self._log(LogLevel.INFO, message, details)

    def warning(self, message: str, details: Optional[dict] = None) -> LogEntry:
        """
        Log a warning message.

        Args:
            message: The log message.
            details: Optional additional details.

        Returns:
            The created LogEntry.
        """
        return self._log(LogLevel.WARNING, message, details)

    def error(self, message: str, details: Optional[dict] = None) -> LogEntry:
        """
        Log an error message.

        Args:
            message: The log message.
            details: Optional additional details.

        Returns:
            The created LogEntry.
        """
        return self._log(LogLevel.ERROR, message, details)

    def success(self, message: str, details: Optional[dict] = None) -> LogEntry:
        """
        Log a success message.

        Args:
            message: The log message.
            details: Optional additional details.

        Returns:
            The created LogEntry.
        """
        return self._log(LogLevel.SUCCESS, message, details)

    def filter_by_level(self, min_level: Optional[LogLevel] = None) -> list[LogEntry]:
        """
        Get log entries filtered by minimum level.

        Args:
            min_level: Minimum log level to include. Uses instance min_level if None.

        Returns:
            List of filtered log entries.
        """
        level = min_level or self._min_level
        level_order = list(LogLevel)
        min_index = level_order.index(level)

        return [entry for entry in self._entries if level_order.index(entry.level) >= min_index]

    def filter_by_levels(self, levels: list[LogLevel]) -> list[LogEntry]:
        """
        Get log entries matching specific levels.

        Args:
            levels: List of log levels to include.

        Returns:
            List of filtered log entries.
        """
        return [entry for entry in self._entries if entry.level in levels]

    def clear(self) -> None:
        """Clear all log entries."""
        self._entries.clear()

    def export_to_file(
        self,
        path: Path | str,
        format: str = "text",
        min_level: Optional[LogLevel] = None,
    ) -> None:
        """
        Export log entries to a file.

        Args:
            path: File path to export to.
            format: Export format - 'text' or 'json'.
            min_level: Minimum log level to include in export.

        Raises:
            ValueError: If format is not supported.
        """
        if isinstance(path, str):
            path = Path(path)

        entries = self.filter_by_level(min_level) if min_level else self._entries

        if format == "text":
            content = "\n".join(entry.format(include_details=True) for entry in entries)
        elif format == "json":
            content = json.dumps(
                [entry.to_dict() for entry in entries],
                indent=2,
                ensure_ascii=False,
            )
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'text' or 'json'.")

        path.write_text(content, encoding="utf-8")


# Singleton instance
_logger_instance: Optional[Logger] = None


def get_logger(max_entries: int = 1000) -> Logger:
    """
    Get the singleton logger instance.

    Args:
        max_entries: Maximum entries (only used on first call).

    Returns:
        The singleton Logger instance.
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = Logger(max_entries=max_entries)
    return _logger_instance
