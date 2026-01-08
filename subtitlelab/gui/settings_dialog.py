from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QWidget,
    QLabel,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QSlider,
    QTextEdit,
    QPushButton,
    QFormLayout,
    QGroupBox,
    QScrollArea,
    QMessageBox,
)
from PyQt6.QtCore import Qt

from .theme import Theme
from .translations import get_translation
from ..core.config import AppConfig


class SettingsDialog(QDialog):
    def __init__(self, config: AppConfig, theme: Theme, parent=None):
        super().__init__(parent)
        self.config = AppConfig.load()
        self.theme = theme
        # Get language from parent if available
        self.lang = getattr(parent, "lang", "en_US") if parent else "en_US"
        self._setup_ui()
        self._load_config()

    def _tr(self, text: str) -> str:
        """Translate text using the current language."""
        return get_translation(text, self.lang)

    def _setup_ui(self):
        self.setWindowTitle(self._tr("Settings"))
        self.setMinimumSize(700, 550)
        self.resize(750, 600)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_llm_tab(), self._tr("🤖 LLM"))
        self.tabs.addTab(self._create_processing_tab(), self._tr("⚙️ Processing"))
        self.tabs.addTab(self._create_prompt_tab(), self._tr("📝 Prompts"))
        self.tabs.addTab(self._create_pricing_tab(), self._tr("💰 Pricing"))
        layout.addWidget(self.tabs)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton(self._tr("Cancel"))
        cancel_btn.setProperty("class", "secondary")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton(self._tr("Save"))
        save_btn.clicked.connect(self._save_config)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

    def _create_llm_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        preset_group = QGroupBox(self._tr("Preset"))
        preset_layout = QFormLayout(preset_group)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(
            ["OpenAI GPT-4o", "OpenAI GPT-4o Mini", "DeepSeek", "Ollama", "Claude", "Custom"]
        )
        self.preset_combo.currentTextChanged.connect(self._on_preset_change)
        preset_layout.addRow(self._tr("Provider:"), self.preset_combo)
        layout.addWidget(preset_group)

        api_group = QGroupBox(self._tr("API Configuration"))
        api_layout = QFormLayout(api_group)

        self.api_base_edit = QLineEdit()
        self.api_base_edit.setPlaceholderText("https://api.openai.com/v1")
        api_layout.addRow(self._tr("API Endpoint:"), self.api_base_edit)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("sk-...")
        api_layout.addRow(self._tr("API Key:"), self.api_key_edit)

        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("gpt-4o")
        api_layout.addRow(self._tr("Model:"), self.model_edit)

        layout.addWidget(api_group)

        params_group = QGroupBox(self._tr("Parameters"))
        params_layout = QFormLayout(params_group)

        self.context_window_spin = QSpinBox()
        self.context_window_spin.setRange(1000, 200000)
        self.context_window_spin.setSingleStep(1000)
        params_layout.addRow(self._tr("Context Window:"), self.context_window_spin)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(100, 32000)
        self.max_tokens_spin.setSingleStep(100)
        params_layout.addRow(self._tr("Max Output Tokens:"), self.max_tokens_spin)

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(10, 600)
        self.timeout_spin.setSuffix(" s")
        params_layout.addRow(self._tr("Timeout:"), self.timeout_spin)

        self.concurrency_spin = QSpinBox()
        self.concurrency_spin.setRange(1, 10)
        params_layout.addRow(self._tr("Concurrency:"), self.concurrency_spin)

        self.json_mode_check = QCheckBox(self._tr("Enable JSON Mode"))
        params_layout.addRow("", self.json_mode_check)

        layout.addWidget(params_group)
        layout.addStretch()

        scroll.setWidget(widget)
        return scroll

    def _create_processing_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        chunk_group = QGroupBox(self._tr("Chunking Strategy"))
        chunk_layout = QFormLayout(chunk_group)

        self.auto_chunk_check = QCheckBox(self._tr("Auto Chunking (Recommended)"))
        self.auto_chunk_check.setChecked(True)
        self.auto_chunk_check.toggled.connect(self._on_auto_chunk_toggle)
        chunk_layout.addRow("", self.auto_chunk_check)

        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(5, 100)
        self.window_size_spin.setEnabled(False)
        chunk_layout.addRow(self._tr("Window Size:"), self.window_size_spin)

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 10)
        self.overlap_spin.setEnabled(False)
        chunk_layout.addRow(self._tr("Window Overlap:"), self.overlap_spin)

        layout.addWidget(chunk_group)

        opt_group = QGroupBox(self._tr("Optimization"))
        opt_layout = QVBoxLayout(opt_group)

        self.semantic_check = QCheckBox(self._tr("Enable Semantic Analysis"))
        opt_layout.addWidget(self.semantic_check)

        self.prefilter_check = QCheckBox(self._tr("Enable Pre-filter"))
        opt_layout.addWidget(self.prefilter_check)

        self.dynamic_window_check = QCheckBox(self._tr("Allow Dynamic Window"))
        opt_layout.addWidget(self.dynamic_window_check)

        layout.addWidget(opt_group)

        quality_group = QGroupBox(self._tr("Quality Scoring"))
        quality_layout = QFormLayout(quality_group)

        self.quality_scoring_check = QCheckBox(self._tr("Enable Quality Scoring (LLM)"))
        quality_layout.addRow("", self.quality_scoring_check)

        self.score_threshold_spin = QDoubleSpinBox()
        self.score_threshold_spin.setRange(0.5, 0.95)
        self.score_threshold_spin.setSingleStep(0.05)
        self.score_threshold_spin.setDecimals(2)
        quality_layout.addRow(self._tr("Score Threshold:"), self.score_threshold_spin)

        layout.addWidget(quality_group)
        layout.addStretch()

        scroll.setWidget(widget)
        return scroll

    def _on_auto_chunk_toggle(self, checked: bool):
        self.window_size_spin.setEnabled(not checked)
        self.overlap_spin.setEnabled(not checked)

    def _create_prompt_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        bg_group = QGroupBox(self._tr("Video Context"))
        bg_layout = QVBoxLayout(bg_group)
        self.background_edit = QTextEdit()
        self.background_edit.setPlaceholderText(
            self._tr(
                "Describe the video content to help AI understand context...\n"
                "Example: Gaming stream by streamer XXX, playing Genshin Impact"
            )
        )
        self.background_edit.setMaximumHeight(100)
        bg_layout.addWidget(self.background_edit)
        layout.addWidget(bg_group)

        terms_group = QGroupBox(self._tr("Terminology List"))
        terms_layout = QVBoxLayout(terms_group)
        self.style_edit = QTextEdit()
        self.style_edit.setPlaceholderText(
            self._tr(
                "List proper nouns, names, and terms for consistent correction...\n"
                "Example: Game=Genshin Impact, Character=Traveler, Streamer=XXX"
            )
        )
        self.style_edit.setMaximumHeight(100)
        terms_layout.addWidget(self.style_edit)
        layout.addWidget(terms_group)

        instr_group = QGroupBox(self._tr("Custom Rules"))
        instr_layout = QVBoxLayout(instr_group)
        self.instructions_edit = QTextEdit()
        self.instructions_edit.setPlaceholderText(
            self._tr(
                "Additional correction rules...\n"
                "Example: Keep all filler words like 'um', 'uh'. Do not delete short pauses."
            )
        )
        self.instructions_edit.setMaximumHeight(100)
        instr_layout.addWidget(self.instructions_edit)
        layout.addWidget(instr_group)

        layout.addStretch()

        scroll.setWidget(widget)
        return scroll

    def _create_pricing_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        info_label = QLabel(
            self._tr("ℹ️ Used for calculating estimated costs based on token usage.")
        )
        info_label.setProperty("class", "secondary")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        pricing_group = QGroupBox(self._tr("Token Pricing"))
        pricing_layout = QFormLayout(pricing_group)

        self.pricing_enabled_check = QCheckBox(self._tr("Enable Cost Estimation"))
        pricing_layout.addRow("", self.pricing_enabled_check)

        self.input_price_spin = QDoubleSpinBox()
        self.input_price_spin.setRange(0, 100)
        self.input_price_spin.setDecimals(4)
        self.input_price_spin.setPrefix("$ ")
        self.input_price_spin.setSuffix(" / 1M tokens")
        pricing_layout.addRow(self._tr("Input Price:"), self.input_price_spin)

        self.output_price_spin = QDoubleSpinBox()
        self.output_price_spin.setRange(0, 100)
        self.output_price_spin.setDecimals(4)
        self.output_price_spin.setPrefix("$ ")
        self.output_price_spin.setSuffix(" / 1M tokens")
        pricing_layout.addRow(self._tr("Output Price:"), self.output_price_spin)

        layout.addWidget(pricing_group)
        layout.addStretch()

        return widget

    def _on_preset_change(self, preset: str):
        presets = {
            "OpenAI GPT-4o": ("https://api.openai.com/v1", "gpt-4o", 128000, 4096),
            "OpenAI GPT-4o Mini": ("https://api.openai.com/v1", "gpt-4o-mini", 128000, 16384),
            "DeepSeek": ("https://api.deepseek.com/v1", "deepseek-chat", 64000, 4096),
            "Ollama": ("http://localhost:11434/v1", "llama3", 8192, 2048),
            "Claude": ("https://api.anthropic.com/v1", "claude-3-opus-20240229", 200000, 4096),
        }

        if preset in presets:
            api_base, model, context, max_tokens = presets[preset]
            self.api_base_edit.setText(api_base)
            self.model_edit.setText(model)
            self.context_window_spin.setValue(context)
            self.max_tokens_spin.setValue(max_tokens)

    def _load_config(self):
        llm = self.config.llm
        self.api_base_edit.setText(llm.api_base)
        self.api_key_edit.setText(llm.api_key)
        self.model_edit.setText(llm.model)
        self.context_window_spin.setValue(llm.context_window)
        self.max_tokens_spin.setValue(llm.max_output_tokens)
        self.timeout_spin.setValue(llm.timeout)
        self.json_mode_check.setChecked(llm.enable_json_mode)

        proc = self.config.processing
        self.concurrency_spin.setValue(proc.concurrency)
        self.window_size_spin.setValue(proc.window_size)
        self.overlap_spin.setValue(proc.window_overlap)
        self.semantic_check.setChecked(proc.enable_semantic_analysis)
        self.prefilter_check.setChecked(proc.enable_pre_filter)
        self.dynamic_window_check.setChecked(proc.allow_dynamic_window)
        self.quality_scoring_check.setChecked(proc.enable_quality_scoring)
        self.score_threshold_spin.setValue(proc.quality_score_threshold)

        auto_chunk = proc.allow_dynamic_window
        self.auto_chunk_check.setChecked(auto_chunk)
        self.window_size_spin.setEnabled(not auto_chunk)
        self.overlap_spin.setEnabled(not auto_chunk)

        prompt = self.config.user_prompt
        self.background_edit.setPlainText(prompt.background_info)
        self.style_edit.setPlainText(prompt.style_guide)
        self.instructions_edit.setPlainText(prompt.custom_instructions)

        pricing = self.config.pricing
        self.pricing_enabled_check.setChecked(pricing.enabled)
        self.input_price_spin.setValue(pricing.input_price)
        self.output_price_spin.setValue(pricing.output_price)

    def _save_config(self):
        try:
            self.config.llm.api_base = self.api_base_edit.text()
            self.config.llm.api_key = self.api_key_edit.text()
            self.config.llm.model = self.model_edit.text()
            self.config.llm.context_window = self.context_window_spin.value()
            self.config.llm.max_output_tokens = self.max_tokens_spin.value()
            self.config.llm.timeout = self.timeout_spin.value()
            self.config.llm.enable_json_mode = self.json_mode_check.isChecked()

            self.config.processing.concurrency = self.concurrency_spin.value()
            self.config.processing.window_size = self.window_size_spin.value()
            self.config.processing.window_overlap = self.overlap_spin.value()
            self.config.processing.enable_semantic_analysis = self.semantic_check.isChecked()
            self.config.processing.enable_pre_filter = self.prefilter_check.isChecked()
            self.config.processing.allow_dynamic_window = self.dynamic_window_check.isChecked()
            self.config.processing.enable_quality_scoring = self.quality_scoring_check.isChecked()
            self.config.processing.quality_score_threshold = self.score_threshold_spin.value()

            self.config.user_prompt.background_info = self.background_edit.toPlainText()
            self.config.user_prompt.style_guide = self.style_edit.toPlainText()
            self.config.user_prompt.custom_instructions = self.instructions_edit.toPlainText()

            self.config.pricing.enabled = self.pricing_enabled_check.isChecked()
            self.config.pricing.input_price = self.input_price_spin.value()
            self.config.pricing.output_price = self.output_price_spin.value()

            self.accept()

        except Exception as e:
            QMessageBox.critical(self, self._tr("Error"), str(e))

    def get_config(self) -> AppConfig:
        return self.config
