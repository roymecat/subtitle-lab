from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, ScrollableContainer
from textual.widgets import (
    Button,
    Label,
    Input,
    Select,
    Switch,
    TabbedContent,
    TabPane,
    TextArea,
    Static,
    Header,
    Footer,
)
from textual.screen import Screen, ModalScreen
from textual.binding import Binding
from textual.reactive import reactive

from ...core.config import AppConfig


class SettingsScreen(ModalScreen):
    """Settings configuration screen."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save"),
    ]

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-container"):
            yield Label("Settings", classes="title")

            with TabbedContent():
                with TabPane("LLM", id="tab-llm"):
                    yield Label("API Base URL")
                    yield Input(self.config.llm.api_base, id="input-api-base")

                    yield Label("API Key")
                    yield Input(self.config.llm.api_key, password=True, id="input-api-key")

                    yield Label("Model")
                    yield Input(self.config.llm.model, id="input-model")

                    yield Label("Context Window")
                    yield Input(
                        str(self.config.llm.context_window), id="input-context", type="integer"
                    )

                with TabPane("Processing", id="tab-processing"):
                    with Horizontal():
                        yield Label("Auto Chunking")
                        yield Switch(
                            value=self.config.processing.allow_dynamic_window,
                            id="switch-auto-chunk",
                        )

                    yield Label("Concurrency")
                    yield Input(
                        str(self.config.processing.concurrency),
                        id="input-concurrency",
                        type="integer",
                    )

                with TabPane("Context", id="tab-context"):
                    yield Label("Video Context")
                    yield TextArea(self.config.user_prompt.background_info, id="input-background")

                    yield Label("Terminology List")
                    yield TextArea(self.config.user_prompt.style_guide, id="input-terminology")

            with Horizontal(classes="dialog-buttons"):
                yield Button("Cancel", variant="default", id="btn-cancel")
                yield Button("Save", variant="primary", id="btn-save")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-save":
            self.save_config()
            self.dismiss(True)
        elif event.button.id == "btn-cancel":
            self.dismiss(False)

    def save_config(self) -> None:
        """Save values back to config object."""
        # LLM
        self.config.llm.api_base = self.query_one("#input-api-base", Input).value
        self.config.llm.api_key = self.query_one("#input-api-key", Input).value
        self.config.llm.model = self.query_one("#input-model", Input).value
        self.config.llm.context_window = int(self.query_one("#input-context", Input).value or 0)

        # Processing
        self.config.processing.allow_dynamic_window = self.query_one(
            "#switch-auto-chunk", Switch
        ).value
        self.config.processing.concurrency = int(
            self.query_one("#input-concurrency", Input).value or 1
        )

        # Prompts
        self.config.user_prompt.background_info = self.query_one("#input-background", TextArea).text
        self.config.user_prompt.style_guide = self.query_one("#input-terminology", TextArea).text

        self.config.save()
