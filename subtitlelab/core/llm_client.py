"""
LLM Client module for SubtitleLab.

Provides OpenAI-compatible API client with intelligent retry mechanism,
concurrency control, token counting, and cost estimation.
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import openai
from openai import AsyncOpenAI

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from .config import LLMConfig, estimate_cost


logger = logging.getLogger(__name__)


class RetryErrorType(str, Enum):
    """Types of errors that can trigger retry logic."""

    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    EMPTY_RESPONSE = "empty_response"
    INVALID_JSON = "invalid_json"
    SERVER_ERROR = "server_error"
    CONNECTION_ERROR = "connection_error"
    UNKNOWN = "unknown"


@dataclass
class RetryAttempt:
    """Record of a single retry attempt."""

    attempt_number: int
    error_type: RetryErrorType
    error_message: str
    wait_time: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class LLMResponse:
    """
    Response from LLM API call.

    Attributes:
        content: The response content (text or parsed JSON)
        raw_content: Raw string content from the API
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        model: Model used for the request
        finish_reason: Reason for completion (stop, length, etc.)
        request_id: OpenAI request ID for debugging
        retries: List of retry attempts made
        total_time: Total time including retries in seconds
        estimated_cost: Estimated cost in USD
    """

    content: Any
    raw_content: str
    input_tokens: int
    output_tokens: int
    model: str
    finish_reason: str
    request_id: Optional[str] = None
    retries: list[RetryAttempt] = field(default_factory=list)
    total_time: float = 0.0
    estimated_cost: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    @property
    def retry_count(self) -> int:
        """Number of retries performed."""
        return len(self.retries)


class SmartRetryHandler:
    """
    Intelligent retry handler with different strategies for different error types.

    Implements:
    - Exponential backoff with jitter
    - Retry-After header respect for rate limits
    - Prompt modification for empty/invalid responses
    - Configurable max attempts per error type
    """

    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter_factor: float = 0.5,
    ):
        """
        Initialize the retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            max_delay: Maximum delay in seconds
            jitter_factor: Factor for random jitter (0-1)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_factor = jitter_factor

        # Error-specific retry configurations
        self._retry_configs = {
            RetryErrorType.RATE_LIMIT: {"multiplier": 2.0, "max_retries": max_retries},
            RetryErrorType.TIMEOUT: {"multiplier": 1.5, "max_retries": max_retries},
            RetryErrorType.EMPTY_RESPONSE: {"multiplier": 1.0, "max_retries": 3},
            RetryErrorType.INVALID_JSON: {"multiplier": 1.0, "max_retries": 3},
            RetryErrorType.SERVER_ERROR: {"multiplier": 2.0, "max_retries": max_retries},
            RetryErrorType.CONNECTION_ERROR: {"multiplier": 1.5, "max_retries": max_retries},
            RetryErrorType.UNKNOWN: {"multiplier": 1.5, "max_retries": 2},
        }

    def classify_error(self, error: Exception) -> RetryErrorType:
        """
        Classify an exception into a retry error type.

        Args:
            error: The exception to classify

        Returns:
            The classified error type
        """
        if isinstance(error, openai.RateLimitError):
            return RetryErrorType.RATE_LIMIT
        elif isinstance(error, openai.APITimeoutError):
            return RetryErrorType.TIMEOUT
        elif isinstance(error, openai.APIConnectionError):
            return RetryErrorType.CONNECTION_ERROR
        elif isinstance(error, openai.APIStatusError):
            if error.status_code >= 500:
                return RetryErrorType.SERVER_ERROR
            return RetryErrorType.UNKNOWN
        elif isinstance(error, EmptyResponseError):
            return RetryErrorType.EMPTY_RESPONSE
        elif isinstance(error, InvalidJSONError):
            return RetryErrorType.INVALID_JSON
        else:
            return RetryErrorType.UNKNOWN

    def should_retry(self, error_type: RetryErrorType, attempt: int) -> bool:
        """
        Determine if a retry should be attempted.

        Args:
            error_type: Type of error encountered
            attempt: Current attempt number (0-based)

        Returns:
            True if retry should be attempted
        """
        config = self._retry_configs.get(error_type, self._retry_configs[RetryErrorType.UNKNOWN])
        return attempt < config["max_retries"]

    def calculate_wait_time(
        self,
        error_type: RetryErrorType,
        attempt: int,
        retry_after: Optional[float] = None,
    ) -> float:
        """
        Calculate wait time before next retry.

        Uses exponential backoff with full jitter strategy.

        Args:
            error_type: Type of error encountered
            attempt: Current attempt number (0-based)
            retry_after: Optional Retry-After header value in seconds

        Returns:
            Wait time in seconds
        """
        # Honor Retry-After header for rate limits
        if retry_after is not None and error_type == RetryErrorType.RATE_LIMIT:
            # Add small jitter to prevent thundering herd
            jitter = random.uniform(0, min(retry_after * 0.1, 5.0))
            return retry_after + jitter

        config = self._retry_configs.get(error_type, self._retry_configs[RetryErrorType.UNKNOWN])
        multiplier = config["multiplier"]

        # Exponential backoff: base_delay * multiplier^attempt
        backoff = self.base_delay * (multiplier**attempt)

        # Cap at max_delay
        backoff = min(backoff, self.max_delay)

        # Full jitter: random value between 0 and backoff
        jitter = random.uniform(0, backoff * self.jitter_factor)

        return backoff + jitter

    def get_retry_after_from_error(self, error: Exception) -> Optional[float]:
        """
        Extract Retry-After header value from an error.

        Args:
            error: The exception to extract from

        Returns:
            Retry-After value in seconds, or None if not available
        """
        if isinstance(error, openai.RateLimitError):
            try:
                retry_after = error.response.headers.get("retry-after")
                if retry_after:
                    # Handle both integer seconds and HTTP-date formats
                    if retry_after.isdigit():
                        return float(retry_after)
                    # For HTTP-date format, default to a reasonable value
                    return 30.0
            except (AttributeError, ValueError):
                pass
        return None

    def modify_prompt_for_retry(
        self,
        error_type: RetryErrorType,
        messages: list[dict],
        attempt: int,
    ) -> list[dict]:
        """
        Modify the prompt for retry based on error type.

        Args:
            error_type: Type of error encountered
            messages: Original messages list
            attempt: Current attempt number

        Returns:
            Modified messages list
        """
        messages = [msg.copy() for msg in messages]

        if error_type == RetryErrorType.EMPTY_RESPONSE:
            # Add instruction to ensure valid output
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n\n请确保输出有效JSON，不要返回空内容。"
            else:
                messages.append({"role": "user", "content": "请确保输出有效JSON，不要返回空内容。"})

        elif error_type == RetryErrorType.INVALID_JSON:
            # Add stricter JSON instruction
            json_instruction = (
                "\n\n重要：请严格按照JSON格式输出，确保：\n"
                "1. 使用双引号包裹字符串\n"
                "2. 不要在JSON前后添加任何额外文字\n"
                "3. 确保JSON语法正确，可以被解析"
            )
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += json_instruction
            else:
                messages.append({"role": "user", "content": json_instruction})

        return messages


class EmptyResponseError(Exception):
    """Raised when the API returns an empty response."""

    pass


class InvalidJSONError(Exception):
    """Raised when the response contains invalid JSON."""

    def __init__(self, message: str, raw_content: str):
        super().__init__(message)
        self.raw_content = raw_content


class TokenCounter:
    """
    Token counter using tiktoken with fallback estimation.

    Provides accurate token counting for supported models and
    heuristic-based estimation for unsupported models.
    """

    # Encoding cache to avoid repeated initialization
    _encoding_cache: dict[str, Any] = {}

    # Model to encoding mapping for common models
    _model_encodings = {
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
    }

    @classmethod
    def count_tokens(cls, text: str, model: str = "gpt-4o") -> int:
        """
        Count tokens in text for a given model.

        Args:
            text: Text to count tokens for
            model: Model name

        Returns:
            Number of tokens
        """
        if not TIKTOKEN_AVAILABLE:
            return cls._estimate_tokens(text)

        try:
            encoding = cls._get_encoding(model)
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Tiktoken failed for model {model}: {e}. Using estimation.")
            return cls._estimate_tokens(text)

    @classmethod
    def count_messages_tokens(
        cls,
        messages: list[dict],
        model: str = "gpt-4o",
    ) -> int:
        """
        Count tokens for a list of chat messages.

        Args:
            messages: List of message dictionaries
            model: Model name

        Returns:
            Estimated total tokens
        """
        if not TIKTOKEN_AVAILABLE:
            total_text = " ".join(str(msg.get("content", "")) for msg in messages)
            return cls._estimate_tokens(total_text)

        try:
            encoding = cls._get_encoding(model)
            num_tokens = 0

            for message in messages:
                # Every message has overhead tokens
                num_tokens += 4  # <|start|>role<|end|>content<|end|>
                for key, value in message.items():
                    if value:
                        num_tokens += len(encoding.encode(str(value)))

            # Every reply is primed with <|start|>assistant<|message|>
            num_tokens += 2

            return num_tokens
        except Exception as e:
            logger.warning(f"Token counting failed: {e}. Using estimation.")
            total_text = " ".join(str(msg.get("content", "")) for msg in messages)
            return cls._estimate_tokens(total_text)

    @classmethod
    def _get_encoding(cls, model: str):
        """Get or create encoding for a model."""
        if model in cls._encoding_cache:
            return cls._encoding_cache[model]

        try:
            # Try to get encoding for the specific model
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fall back to a default encoding
            # Try to match partial model names
            for model_prefix, encoding_name in cls._model_encodings.items():
                if model_prefix in model.lower():
                    encoding = tiktoken.get_encoding(encoding_name)
                    break
            else:
                # Default to o200k_base (GPT-4o encoding)
                encoding = tiktoken.get_encoding("o200k_base")

        cls._encoding_cache[model] = encoding
        return encoding

    @classmethod
    def _estimate_tokens(cls, text: str) -> int:
        """
        Estimate token count using heuristics.

        Uses multiple estimation methods and returns the higher value
        for safety in cost/limit calculations.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Method 1: Character-based (English ~4 chars/token, CJK ~1.5 chars/token)
        # Detect if text is primarily CJK
        cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        total_chars = len(text)

        if cjk_count > total_chars * 0.3:
            # Primarily CJK text
            char_estimate = int(total_chars / 1.5)
        else:
            # Primarily English/Latin text
            char_estimate = total_chars // 4

        # Method 2: Word-based (~1.3 tokens per word for English)
        word_count = len(text.split())
        word_estimate = int(word_count * 1.3)

        # Return the higher estimate for safety
        return max(char_estimate, word_estimate, 1)


class LLMClient:
    """
    OpenAI-compatible API client with intelligent retry mechanism.

    Features:
    - Async-first design using AsyncOpenAI
    - Configurable concurrency via semaphore
    - Smart retry with different strategies per error type
    - Token counting and cost estimation
    - Response validation
    """

    def __init__(
        self,
        config: LLMConfig,
        concurrency: int = 3,
        on_retry: Optional[Callable[[RetryAttempt], None]] = None,
    ):
        """
        Initialize the LLM client.

        Args:
            config: LLM configuration
            concurrency: Maximum concurrent requests
            on_retry: Optional callback for retry events
        """
        self.config = config
        self._semaphore = asyncio.Semaphore(concurrency)
        self._on_retry = on_retry

        # Initialize AsyncOpenAI client
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
            timeout=config.timeout,
            max_retries=0,  # We handle retries ourselves
        )

        # Initialize retry handler
        self._retry_handler = SmartRetryHandler(
            max_retries=config.max_retries,
            base_delay=config.retry_delay,
        )

    async def chat_completion(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: Optional[bool] = None,
        validate_json: bool = True,
        required_fields: Optional[list[str]] = None,
    ) -> LLMResponse:
        """
        Make a chat completion request with retry logic.

        Args:
            messages: List of message dictionaries
            temperature: Override temperature (uses config default if None)
            max_tokens: Override max tokens (uses config default if None)
            json_mode: Override JSON mode (uses config default if None)
            validate_json: Whether to validate JSON in response
            required_fields: Required fields to check in JSON response

        Returns:
            LLMResponse with content and metadata

        Raises:
            openai.APIError: If all retries are exhausted
            InvalidJSONError: If JSON validation fails after retries
        """
        async with self._semaphore:
            return await self._execute_with_retry(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                validate_json=validate_json,
                required_fields=required_fields,
            )

    async def _execute_with_retry(
        self,
        messages: list[dict],
        temperature: Optional[float],
        max_tokens: Optional[int],
        json_mode: Optional[bool],
        validate_json: bool,
        required_fields: Optional[list[str]],
    ) -> LLMResponse:
        """Execute request with retry logic."""
        start_time = time.time()
        retries: list[RetryAttempt] = []
        current_messages = messages

        attempt = 0
        last_error: Optional[Exception] = None

        while True:
            try:
                response = await self._make_request(
                    messages=current_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )

                # Validate response
                content = self._validate_response(
                    response=response,
                    validate_json=validate_json,
                    required_fields=required_fields,
                )

                # Build successful response
                total_time = time.time() - start_time
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0

                return LLMResponse(
                    content=content,
                    raw_content=response.choices[0].message.content or "",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=response.model,
                    finish_reason=response.choices[0].finish_reason or "unknown",
                    request_id=response.id,
                    retries=retries,
                    total_time=total_time,
                    estimated_cost=estimate_cost(self.config.model, input_tokens, output_tokens),
                )

            except Exception as e:
                last_error = e
                error_type = self._retry_handler.classify_error(e)

                # Check if we should retry
                if not self._retry_handler.should_retry(error_type, attempt):
                    logger.error(f"Max retries exceeded for {error_type.value}: {e}")
                    raise

                # Calculate wait time
                retry_after = self._retry_handler.get_retry_after_from_error(e)
                wait_time = self._retry_handler.calculate_wait_time(
                    error_type, attempt, retry_after
                )

                # Record retry attempt
                retry_attempt = RetryAttempt(
                    attempt_number=attempt + 1,
                    error_type=error_type,
                    error_message=str(e),
                    wait_time=wait_time,
                )
                retries.append(retry_attempt)

                # Log retry
                logger.warning(
                    f"Retry {attempt + 1} for {error_type.value}: {e}. Waiting {wait_time:.2f}s"
                )

                # Notify callback
                if self._on_retry:
                    self._on_retry(retry_attempt)

                # Modify prompt if needed
                current_messages = self._retry_handler.modify_prompt_for_retry(
                    error_type, current_messages, attempt
                )

                # Wait before retry
                await asyncio.sleep(wait_time)
                attempt += 1

    async def _make_request(
        self,
        messages: list[dict],
        temperature: Optional[float],
        max_tokens: Optional[int],
        json_mode: Optional[bool],
    ):
        """Make the actual API request."""
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_output_tokens,
        }

        # Add JSON mode if enabled
        use_json_mode = json_mode if json_mode is not None else self.config.enable_json_mode
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        return await self._client.chat.completions.create(**kwargs)

    def _validate_response(
        self,
        response,
        validate_json: bool,
        required_fields: Optional[list[str]],
    ) -> Any:
        """
        Validate the API response.

        Args:
            response: The API response object
            validate_json: Whether to validate JSON
            required_fields: Required fields in JSON response

        Returns:
            Parsed content (dict if JSON, str otherwise)

        Raises:
            EmptyResponseError: If response is empty
            InvalidJSONError: If JSON validation fails
        """
        if not response.choices:
            raise EmptyResponseError("No choices in response")

        content = response.choices[0].message.content

        if not content or not content.strip():
            raise EmptyResponseError("Empty content in response")

        if not validate_json:
            return content

        # Try to parse JSON
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown code blocks
            extracted = self._extract_json_from_content(content)
            if extracted:
                try:
                    parsed = json.loads(extracted)
                except json.JSONDecodeError:
                    raise InvalidJSONError(
                        f"Invalid JSON in response: {e}",
                        raw_content=content,
                    )
            else:
                raise InvalidJSONError(
                    f"Invalid JSON in response: {e}",
                    raw_content=content,
                )

        # Check required fields
        if required_fields:
            missing = [f for f in required_fields if f not in parsed]
            if missing:
                raise InvalidJSONError(
                    f"Missing required fields: {missing}",
                    raw_content=content,
                )

        return parsed

    def _extract_json_from_content(self, content: str) -> Optional[str]:
        """
        Try to extract JSON from content that may be wrapped in markdown.

        Args:
            content: Raw content string

        Returns:
            Extracted JSON string or None
        """
        content = content.strip()

        # Try to find JSON in code blocks
        import re

        # Match ```json ... ``` or ``` ... ```
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1).strip()

        # Try to find JSON object or array directly
        # Look for content starting with { or [
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start_idx = content.find(start_char)
            if start_idx != -1:
                # Find matching end
                depth = 0
                for i, char in enumerate(content[start_idx:], start_idx):
                    if char == start_char:
                        depth += 1
                    elif char == end_char:
                        depth -= 1
                        if depth == 0:
                            return content[start_idx : i + 1]

        return None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return TokenCounter.count_tokens(text, self.config.model)

    def count_messages_tokens(self, messages: list[dict]) -> int:
        """
        Count tokens for messages.

        Args:
            messages: List of message dictionaries

        Returns:
            Estimated total tokens
        """
        return TokenCounter.count_messages_tokens(messages, self.config.model)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        return estimate_cost(self.config.model, input_tokens, output_tokens)

    async def close(self) -> None:
        """Close the client and release resources."""
        await self._client.close()

    async def __aenter__(self) -> "LLMClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
