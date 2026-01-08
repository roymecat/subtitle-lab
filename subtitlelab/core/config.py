"""Configuration management for SubtitleLab."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


CONFIG_DIR = Path.home() / ".subtitlelab"
CONFIG_FILE = CONFIG_DIR / "config.json"


@dataclass
class LLMConfig:
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    context_window: int = 128000
    max_output_tokens: int = 16384
    temperature: float = 0.3
    timeout: int = 120
    max_retries: int = 5
    retry_delay: float = 1.0
    enable_json_mode: bool = True
    enable_streaming: bool = False

    PRESETS: dict = field(
        default_factory=lambda: {
            "openai-gpt4o": {
                "api_base": "https://api.openai.com/v1",
                "model": "gpt-4o",
                "context_window": 128000,
                "max_output_tokens": 16384,
            },
            "openai-gpt4o-mini": {
                "api_base": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "context_window": 128000,
                "max_output_tokens": 16384,
            },
            "deepseek": {
                "api_base": "https://api.deepseek.com",
                "model": "deepseek-chat",
                "context_window": 64000,
                "max_output_tokens": 8000,
            },
            "ollama": {
                "api_base": "http://localhost:11434/v1",
                "model": "llama3.2",
                "context_window": 128000,
                "max_output_tokens": 4096,
            },
            "claude": {
                "api_base": "https://api.anthropic.com/v1",
                "model": "claude-3-5-sonnet-20241022",
                "context_window": 200000,
                "max_output_tokens": 8192,
            },
        },
        repr=False,
    )

    def apply_preset(self, preset_name: str) -> None:
        if preset_name in self.PRESETS:
            preset = self.PRESETS[preset_name]
            for key, value in preset.items():
                if hasattr(self, key):
                    setattr(self, key, value)


@dataclass
class ProcessingConfig:
    window_size: int = 20
    window_overlap: int = 3
    min_window_size: int = 10
    max_window_size: int = 50
    allow_dynamic_window: bool = True
    enable_semantic_analysis: bool = True
    enable_pre_filter: bool = True
    concurrency: int = 3
    max_subtitle_duration: float = 6.0
    min_subtitle_duration: float = 0.5
    scene_gap_threshold: float = 2.0
    enable_quality_scoring: bool = False
    quality_score_threshold: float = 0.7
    quality_score_max_retries: int = 2


@dataclass
class UserPromptConfig:
    background_info: str = ""
    character_names: dict[str, str] = field(default_factory=dict)
    custom_instructions: str = ""
    style_guide: str = ""

    def to_prompt_section(self) -> str:
        parts = []
        if self.background_info:
            parts.append(f"## 背景信息\n{self.background_info}")
        if self.character_names:
            names_str = "\n".join(
                f"- {wrong} → {correct}" for wrong, correct in self.character_names.items()
            )
            parts.append(f"## 人名对照表\n{names_str}")
        if self.style_guide:
            parts.append(f"## 风格指南\n{self.style_guide}")
        if self.custom_instructions:
            parts.append(f"## 额外指令\n{self.custom_instructions}")
        return "\n\n".join(parts)


@dataclass
class UIConfig:
    theme_mode: str = "dark"
    window_width: int = 1400
    window_height: int = 900
    font_size: int = 14
    show_original_text: bool = True
    auto_scroll_log: bool = True
    language: str = "zh"


@dataclass
class PricingConfig:
    """User-configurable API pricing. Prices are per 1M tokens in USD."""

    input_price: float = 0.0
    output_price: float = 0.0
    enabled: bool = False

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        if not self.enabled:
            return 0.0
        input_cost = (input_tokens / 1_000_000) * self.input_price
        output_cost = (output_tokens / 1_000_000) * self.output_price
        return input_cost + output_cost


MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        for model_prefix, prices in MODEL_PRICING.items():
            if model_prefix in model.lower():
                pricing = prices
                break
    if not pricing:
        return 0.0
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    user_prompt: UserPromptConfig = field(default_factory=UserPromptConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    recent_files: list[str] = field(default_factory=list)
    max_recent_files: int = 10

    def add_recent_file(self, file_path: str) -> None:
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)
        if len(self.recent_files) > self.max_recent_files:
            self.recent_files = self.recent_files[: self.max_recent_files]

    def save(self, path: Optional[Path] = None) -> None:
        save_path = path or CONFIG_FILE
        save_path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = {
            "llm": asdict(self.llm),
            "processing": asdict(self.processing),
            "user_prompt": asdict(self.user_prompt),
            "ui": asdict(self.ui),
            "pricing": asdict(self.pricing),
            "recent_files": self.recent_files,
        }
        if "PRESETS" in config_dict["llm"]:
            del config_dict["llm"]["PRESETS"]
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppConfig":
        load_path = path or CONFIG_FILE
        if not load_path.exists():
            return cls()
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            return cls(
                llm=LLMConfig(**config_dict.get("llm", {})),
                processing=ProcessingConfig(**config_dict.get("processing", {})),
                user_prompt=UserPromptConfig(**config_dict.get("user_prompt", {})),
                ui=UIConfig(**config_dict.get("ui", {})),
                pricing=PricingConfig(**config_dict.get("pricing", {})),
                recent_files=config_dict.get("recent_files", []),
            )
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"Warning: Failed to load config, using defaults: {e}")
            return cls()

    @classmethod
    def get_default(cls) -> "AppConfig":
        return cls()
