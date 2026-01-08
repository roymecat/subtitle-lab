"""Core processing modules for SubtitleLab."""

from .models import (
    SubtitleEntry,
    ProcessedEntry,
    ProcessingAction,
    BatchResult,
    ProcessingStats,
)
from .config import AppConfig, LLMConfig, ProcessingConfig
from .preprocessor import (
    HallucinationDetector,
    HallucinationType,
    DetectionResult,
    FilteredEntry,
    FilterStats,
    PreFilterResult,
    PreFilter,
    create_default_prefilter,
)
from .llm_client import (
    LLMClient,
    LLMResponse,
    SmartRetryHandler,
    TokenCounter,
    RetryErrorType,
    RetryAttempt,
    EmptyResponseError,
    InvalidJSONError,
)
from .chunker import SmartChunker, ChunkerConfig, SlidingWindowManager, Batch
from .semantic_analyzer import SemanticAnalyzer
from .processor import SubtitleProcessor, ProcessorCallbacks, ProcessorState, SYSTEM_PROMPT
from .speaker_detector import SpeakerDetector, SpeakerInfo, SpeakerSegment
from .quality_validator import QualityValidator, ValidationResult, ValidationIssue
from .quality_scorer import QualityScorer, ScoringResult

__all__ = [
    # Models
    "SubtitleEntry",
    "ProcessedEntry",
    "ProcessingAction",
    "BatchResult",
    "ProcessingStats",
    # Config
    "AppConfig",
    "LLMConfig",
    "ProcessingConfig",
    # Preprocessor
    "HallucinationDetector",
    "HallucinationType",
    "DetectionResult",
    "FilteredEntry",
    "FilterStats",
    "PreFilterResult",
    "PreFilter",
    "create_default_prefilter",
    # LLM Client
    "LLMClient",
    "LLMResponse",
    "SmartRetryHandler",
    "TokenCounter",
    "RetryErrorType",
    "RetryAttempt",
    "EmptyResponseError",
    "InvalidJSONError",
    # Chunker
    "SmartChunker",
    "ChunkerConfig",
    "SlidingWindowManager",
    "Batch",
    # Semantic Analyzer
    "SemanticAnalyzer",
    # Processor
    "SubtitleProcessor",
    "ProcessorCallbacks",
    "ProcessorState",
    "SYSTEM_PROMPT",
    # Speaker Detector
    "SpeakerDetector",
    "SpeakerInfo",
    "SpeakerSegment",
    # Quality Validator
    "QualityValidator",
    "ValidationResult",
    "ValidationIssue",
    # Quality Scorer
    "QualityScorer",
    "ScoringResult",
]
