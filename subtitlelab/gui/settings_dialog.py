from typing import Callable
import flet as ft
from .theme import Theme
from ..core.config import AppConfig, LLMConfig


class SettingsDialog(ft.AlertDialog):
    def __init__(self, config: AppConfig, theme: Theme, on_save: Callable):
        super().__init__()
        self.config = config
        self.theme = theme
        self._on_save_callback = on_save
        self.modal = True
        self.title = ft.Text(
            "Settings", size=20, weight=ft.FontWeight.BOLD, color=theme.text_primary
        )
        self.surface_color = theme.surface
        self.bgcolor = theme.surface

        # --- State Holders ---
        self.llm_refs = {}
        self.proc_refs = {}
        self.prompt_refs = {}
        self.pricing_refs = {}
        self.char_rows = []

        self.content = self._build_content()
        self.actions = self._build_actions()
        self.actions_alignment = ft.MainAxisAlignment.END

    def _build_content(self):
        self.tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tabs=[
                ft.Tab(
                    text="LLM Configuration",
                    icon="smart_toy_outlined",
                    content=self._build_llm_tab(),
                ),
                ft.Tab(
                    text="Processing",
                    icon="memory_outlined",
                    content=self._build_processing_tab(),
                ),
                ft.Tab(
                    text="Prompts",
                    icon="edit_note_outlined",
                    content=self._build_prompt_tab(),
                ),
                ft.Tab(
                    text="Pricing",
                    icon="attach_money_outlined",
                    content=self._build_pricing_tab(),
                ),
            ],
            expand=True,
            divider_color=self.theme.border,
            indicator_color=self.theme.primary,
            label_color=self.theme.primary,
            unselected_label_color=self.theme.text_secondary,
        )

        return ft.Container(
            content=self.tabs,
            width=800,
            height=600,
            bgcolor=self.theme.surface,
        )

    def _input_style(self):
        return {
            "border_color": self.theme.border,
            "text_size": 14,
            "color": self.theme.text_primary,
            "cursor_color": self.theme.primary,
            "focused_border_color": self.theme.primary,
            "label_style": ft.TextStyle(color=self.theme.text_secondary),
        }

    def _build_llm_tab(self):
        # Refs
        self.llm_refs["preset"] = ft.Dropdown(
            label="Preset",
            options=[
                ft.dropdown.Option("openai-gpt4o", "OpenAI GPT-4o"),
                ft.dropdown.Option("openai-gpt4o-mini", "OpenAI GPT-4o Mini"),
                ft.dropdown.Option("deepseek", "DeepSeek"),
                ft.dropdown.Option("ollama", "Ollama"),
                ft.dropdown.Option("claude", "Claude"),
                ft.dropdown.Option("custom", "Custom"),
            ],
            value="custom",  # Default, logic to detect later
            on_change=self._on_preset_change,
            **self._input_style(),
        )
        self.llm_refs["api_base"] = ft.TextField(
            label="API Endpoint URL", value=self.config.llm.api_base, **self._input_style()
        )
        self.llm_refs["api_key"] = ft.TextField(
            label="API Key",
            password=True,
            can_reveal_password=True,
            value=self.config.llm.api_key,
            **self._input_style(),
        )
        self.llm_refs["model"] = ft.TextField(
            label="Model Name", value=self.config.llm.model, **self._input_style()
        )
        self.llm_refs["context_window"] = ft.TextField(
            label="Context Window",
            value=str(self.config.llm.context_window),
            keyboard_type=ft.KeyboardType.NUMBER,
            **self._input_style(),
        )
        self.llm_refs["max_tokens"] = ft.TextField(
            label="Max Output Tokens",
            value=str(self.config.llm.max_output_tokens),
            keyboard_type=ft.KeyboardType.NUMBER,
            **self._input_style(),
        )
        self.llm_refs["timeout"] = ft.TextField(
            label="Timeout (s)",
            value=str(self.config.llm.timeout),
            keyboard_type=ft.KeyboardType.NUMBER,
            **self._input_style(),
        )

        self.llm_refs["concurrency_label"] = ft.Text(
            f"Concurrency: {self.config.processing.concurrency}", color=self.theme.text_primary
        )
        self.llm_refs["concurrency"] = ft.Slider(
            min=1,
            max=10,
            divisions=9,
            value=self.config.processing.concurrency,
            active_color=self.theme.primary,
            on_change=lambda e: setattr(
                self.llm_refs["concurrency_label"], "value", f"Concurrency: {int(e.control.value)}"
            )
            or self.llm_refs["concurrency_label"].update(),
        )

        self.llm_refs["json_mode"] = ft.Switch(
            label="Enable JSON Mode",
            value=self.config.llm.enable_json_mode,
            active_color=self.theme.primary,
        )

        # Test Connection Button
        test_btn = ft.ElevatedButton(
            "Test Connection",
            icon="network_check",
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=self.theme.success,
                shape=ft.RoundedRectangleBorder(radius=self.theme.radius.SM),
            ),
            on_click=self._test_connection,
        )

        return ft.Container(
            content=ft.Column(
                [
                    self.llm_refs["preset"],
                    ft.Divider(color=self.theme.border),
                    self.llm_refs["api_base"],
                    self.llm_refs["api_key"],
                    self.llm_refs["model"],
                    ft.Row(
                        [
                            ft.Container(content=self.llm_refs["context_window"], expand=True),
                            ft.Container(content=self.llm_refs["max_tokens"], expand=True),
                            ft.Container(content=self.llm_refs["timeout"], expand=True),
                        ],
                        spacing=20,
                    ),
                    ft.Divider(color=self.theme.border),
                    self.llm_refs["concurrency_label"],
                    self.llm_refs["concurrency"],
                    ft.Row([self.llm_refs["json_mode"], ft.Container(expand=True), test_btn]),
                ],
                scroll=ft.ScrollMode.AUTO,
                spacing=15,
            ),
            padding=20,
        )

    def _build_processing_tab(self):
        pc = self.config.processing

        self.proc_refs["window_size_label"] = ft.Text(
            f"Window Size: {pc.window_size}", color=self.theme.text_primary
        )
        self.proc_refs["window_size"] = ft.Slider(
            min=pc.min_window_size,
            max=pc.max_window_size,
            divisions=pc.max_window_size - pc.min_window_size,
            value=pc.window_size,
            active_color=self.theme.primary,
            on_change=lambda e: setattr(
                self.proc_refs["window_size_label"], "value", f"Window Size: {int(e.control.value)}"
            )
            or self.proc_refs["window_size_label"].update(),
        )

        self.proc_refs["overlap_label"] = ft.Text(
            f"Window Overlap: {pc.window_overlap}", color=self.theme.text_primary
        )
        self.proc_refs["overlap"] = ft.Slider(
            min=0,
            max=10,
            divisions=10,
            value=pc.window_overlap,
            active_color=self.theme.primary,
            on_change=lambda e: setattr(
                self.proc_refs["overlap_label"], "value", f"Window Overlap: {int(e.control.value)}"
            )
            or self.proc_refs["overlap_label"].update(),
        )

        self.proc_refs["semantic"] = ft.Switch(
            label="Enable Semantic Analysis",
            value=pc.enable_semantic_analysis,
            active_color=self.theme.primary,
        )
        self.proc_refs["prefilter"] = ft.Switch(
            label="Enable Pre-filter", value=pc.enable_pre_filter, active_color=self.theme.primary
        )
        self.proc_refs["dynamic_window"] = ft.Switch(
            label="Allow Dynamic Window",
            value=pc.allow_dynamic_window,
            active_color=self.theme.primary,
        )

        self.proc_refs["quality_scoring"] = ft.Switch(
            label="Enable Quality Scoring (LLM)",
            value=pc.enable_quality_scoring,
            active_color=self.theme.primary,
        )

        self.proc_refs["score_threshold_label"] = ft.Text(
            f"Score Threshold: {pc.quality_score_threshold:.1f}",
            color=self.theme.text_primary,
        )
        self.proc_refs["score_threshold"] = ft.Slider(
            min=0.5,
            max=0.95,
            divisions=9,
            value=pc.quality_score_threshold,
            active_color=self.theme.primary,
            on_change=lambda e: setattr(
                self.proc_refs["score_threshold_label"],
                "value",
                f"Score Threshold: {e.control.value:.1f}",
            )
            or self.proc_refs["score_threshold_label"].update(),
        )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        "Chunking Strategy",
                        size=16,
                        weight=ft.FontWeight.BOLD,
                        color=self.theme.text_primary,
                    ),
                    self.proc_refs["window_size_label"],
                    self.proc_refs["window_size"],
                    self.proc_refs["overlap_label"],
                    self.proc_refs["overlap"],
                    ft.Divider(color=self.theme.border),
                    ft.Text(
                        "Optimization",
                        size=16,
                        weight=ft.FontWeight.BOLD,
                        color=self.theme.text_primary,
                    ),
                    self.proc_refs["semantic"],
                    self.proc_refs["prefilter"],
                    self.proc_refs["dynamic_window"],
                    ft.Divider(color=self.theme.border),
                    ft.Text(
                        "Quality Scoring",
                        size=16,
                        weight=ft.FontWeight.BOLD,
                        color=self.theme.text_primary,
                    ),
                    self.proc_refs["quality_scoring"],
                    self.proc_refs["score_threshold_label"],
                    self.proc_refs["score_threshold"],
                ],
                scroll=ft.ScrollMode.AUTO,
                spacing=20,
            ),
            padding=20,
        )

    def _build_prompt_tab(self):
        up = self.config.user_prompt

        self.prompt_refs["background"] = ft.TextField(
            label="Background Information",
            multiline=True,
            min_lines=3,
            max_lines=5,
            value=up.background_info,
            hint_text="Context about the video content...",
            **self._input_style(),
        )

        self.prompt_refs["style"] = ft.TextField(
            label="Style Guide",
            multiline=True,
            min_lines=3,
            value=up.style_guide,
            hint_text="Translation tone, specific terminology...",
            **self._input_style(),
        )

        self.prompt_refs["instructions"] = ft.TextField(
            label="Custom Instructions",
            multiline=True,
            min_lines=3,
            value=up.custom_instructions,
            hint_text="Any additional prompt instructions...",
            **self._input_style(),
        )

        # Character Names Editor
        self.char_list = ft.Column(spacing=10)
        for wrong, correct in up.character_names.items():
            self._add_char_row(wrong, correct)

        add_char_btn = ft.OutlinedButton(
            "Add Character Pair",
            icon="add",
            style=ft.ButtonStyle(color=self.theme.primary),
            on_click=lambda e: self._add_char_row(),
        )

        return ft.Container(
            content=ft.Column(
                [
                    self.prompt_refs["background"],
                    ft.Divider(color=self.theme.border),
                    ft.Text(
                        "Character Names (Find -> Replace)",
                        weight=ft.FontWeight.BOLD,
                        color=self.theme.text_primary,
                    ),
                    self.char_list,
                    add_char_btn,
                    ft.Divider(color=self.theme.border),
                    self.prompt_refs["style"],
                    self.prompt_refs["instructions"],
                ],
                scroll=ft.ScrollMode.AUTO,
                spacing=15,
            ),
            padding=20,
        )

    def _add_char_row(self, wrong="", correct=""):
        row_ref = {}

        def delete_row(e):
            self.char_list.controls.remove(row)
            self.char_rows.remove(row_ref)
            self.char_list.update()

        w_input = ft.TextField(
            value=wrong,
            label="Original Name",
            expand=True,
            height=40,
            content_padding=10,
            **self._input_style(),
        )
        c_input = ft.TextField(
            value=correct,
            label="Correct Name",
            expand=True,
            height=40,
            content_padding=10,
            **self._input_style(),
        )
        del_btn = ft.IconButton(
            icon="delete_outline", icon_color=self.theme.error, on_click=delete_row
        )

        row = ft.Row([w_input, c_input, del_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
        row_ref.update({"wrong": w_input, "correct": c_input, "row": row})

        self.char_rows.append(row_ref)
        self.char_list.controls.append(row)
        if self.page:
            self.char_list.update()

    def _build_pricing_tab(self):
        pc = self.config.pricing

        self.pricing_refs["enabled"] = ft.Switch(
            label="Enable Cost Estimation", value=pc.enabled, active_color=self.theme.primary
        )

        self.pricing_refs["input"] = ft.TextField(
            label="Input Price ($ per 1M tokens)",
            value=str(pc.input_price),
            keyboard_type=ft.KeyboardType.NUMBER,
            prefix_text="$",
            **self._input_style(),
        )

        self.pricing_refs["output"] = ft.TextField(
            label="Output Price ($ per 1M tokens)",
            value=str(pc.output_price),
            keyboard_type=ft.KeyboardType.NUMBER,
            prefix_text="$",
            **self._input_style(),
        )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.Icon("info_outline", color=self.theme.secondary),
                                ft.Text(
                                    "Used for calculating estimated costs based on token usage.",
                                    color=self.theme.text_secondary,
                                    size=12,
                                ),
                            ]
                        ),
                        bgcolor=self.theme.surface_light,
                        padding=10,
                        border_radius=self.theme.radius.SM,
                    ),
                    self.pricing_refs["enabled"],
                    self.pricing_refs["input"],
                    self.pricing_refs["output"],
                ],
                scroll=ft.ScrollMode.AUTO,
                spacing=20,
            ),
            padding=20,
        )

    def _build_actions(self):
        return [
            ft.TextButton(
                "Cancel",
                style=ft.ButtonStyle(color=self.theme.text_secondary),
                on_click=self._cancel,
            ),
            ft.ElevatedButton(
                "Save Changes",
                style=ft.ButtonStyle(
                    color=ft.Colors.WHITE,
                    bgcolor=self.theme.primary,
                    shape=ft.RoundedRectangleBorder(radius=self.theme.radius.SM),
                ),
                on_click=self._save,
            ),
        ]

    def _on_preset_change(self, e):
        preset_key = e.control.value
        presets = self.config.llm.PRESETS
        if preset_key in presets:
            preset = presets[preset_key]
            self.llm_refs["api_base"].value = preset.get("api_base", "")
            self.llm_refs["model"].value = preset.get("model", "")
            self.llm_refs["context_window"].value = str(preset.get("context_window", ""))
            self.llm_refs["max_tokens"].value = str(preset.get("max_output_tokens", ""))
            self.llm_refs["api_base"].update()
            self.llm_refs["model"].update()
            self.llm_refs["context_window"].update()
            self.llm_refs["max_tokens"].update()

    def _test_connection(self, e):
        # Implementation of a simple connection test
        # In a real app, this would use the LLMClient
        # For now, we simulate visual feedback
        e.control.text = "Testing..."
        e.control.disabled = True
        e.control.update()

        # We can't easily do async sleep here without asyncio,
        # but let's assume we just reset it for UI feedback
        # Or better, show a snackbar (if page is available)

        if self.page:
            self.page.show_snack_bar(ft.SnackBar(content=ft.Text("Connection test initiated...")))

        e.control.text = "Test Connection"
        e.control.disabled = False
        e.control.update()

    def _save(self, e):
        try:
            # Update LLM Config
            self.config.llm.api_base = self.llm_refs["api_base"].value
            self.config.llm.api_key = self.llm_refs["api_key"].value
            self.config.llm.model = self.llm_refs["model"].value
            self.config.llm.context_window = int(self.llm_refs["context_window"].value or 0)
            self.config.llm.max_output_tokens = int(self.llm_refs["max_tokens"].value or 0)
            self.config.llm.timeout = int(self.llm_refs["timeout"].value or 60)
            self.config.llm.enable_json_mode = self.llm_refs["json_mode"].value

            # Update Processing Config (Concurrency is in ProcessingConfig, but was on LLM tab)
            self.config.processing.concurrency = int(self.llm_refs["concurrency"].value)
            self.config.processing.window_size = int(self.proc_refs["window_size"].value)
            self.config.processing.window_overlap = int(self.proc_refs["overlap"].value)
            self.config.processing.enable_semantic_analysis = self.proc_refs["semantic"].value
            self.config.processing.enable_pre_filter = self.proc_refs["prefilter"].value
            self.config.processing.allow_dynamic_window = self.proc_refs["dynamic_window"].value
            self.config.processing.enable_quality_scoring = self.proc_refs["quality_scoring"].value
            self.config.processing.quality_score_threshold = float(
                self.proc_refs["score_threshold"].value
            )

            # Update Prompt Config
            self.config.user_prompt.background_info = self.prompt_refs["background"].value
            self.config.user_prompt.style_guide = self.prompt_refs["style"].value
            self.config.user_prompt.custom_instructions = self.prompt_refs["instructions"].value

            # Reconstruct character names dict
            new_chars = {}
            for row in self.char_rows:
                wrong = row["wrong"].value
                correct = row["correct"].value
                if wrong and correct:
                    new_chars[wrong] = correct
            self.config.user_prompt.character_names = new_chars

            # Update Pricing Config
            self.config.pricing.enabled = self.pricing_refs["enabled"].value
            self.config.pricing.input_price = float(self.pricing_refs["input"].value or 0.0)
            self.config.pricing.output_price = float(self.pricing_refs["output"].value or 0.0)

            # Call save callback
            self._on_save_callback(self.config)
            self.open = False
            self.page.update()

        except ValueError as ex:
            if self.page:
                self.page.show_snack_bar(
                    ft.SnackBar(
                        content=ft.Text(f"Error saving settings: {str(ex)}"),
                        bgcolor=self.theme.error,
                    )
                )

    def _cancel(self, e):
        self.open = False
        self.page.update()
