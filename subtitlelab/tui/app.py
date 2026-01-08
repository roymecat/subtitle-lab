from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Button, DataTable, Log, ProgressBar, Static, Label
from textual.screen import Screen
from textual.binding import Binding
from textual.reactive import reactive
from textual import work
from textual.worker import Worker, WorkerState

from pathlib import Path
import asyncio
from datetime import datetime

from ..core.config import AppConfig
from ..core.processor import SubtitleProcessor, ProcessorCallbacks
from ..core.models import BatchResult
from .screens.settings import SettingsScreen


class Dashboard(Screen):
    BINDINGS = [
        Binding("ctrl+o", "import_file", "Import File"),
        Binding("ctrl+s", "start_processing", "Start Processing"),
        Binding("ctrl+c", "cancel_processing", "Cancel Processing"),
        Binding("ctrl+e", "export_file", "Export File"),
        Binding("ctrl+comma", "open_settings", "Settings"),
    ]

    current_file: reactive[Path | None] = reactive(None)
    processor: SubtitleProcessor | None = None
    start_time: datetime | None = None

    def compose(self) -> ComposeResult:
        with Horizontal():
            # Sidebar
            with Vertical(id="sidebar"):
                yield Label("ACTIONS", classes="title")
                yield Button("Import File", id="btn-import", variant="primary")
                yield Button("Start Processing", id="btn-start", variant="success", disabled=True)
                yield Button("Cancel", id="btn-cancel", variant="error", disabled=True)
                yield Button("Export", id="btn-export", disabled=True)
                yield Static()  # Spacer
                yield Button("Settings", id="btn-settings")

            # Main Content
            with Vertical(id="main-content"):
                yield DataTable(id="subtitle-table")

                # Bottom Panel
                with Horizontal(classes="stats-container"):
                    with Vertical(classes="stat-card"):
                        yield Label("TOTAL", classes="stat-label")
                        yield Label("0", id="stat-total", classes="stat-value")
                    with Vertical(classes="stat-card"):
                        yield Label("PROCESSED", classes="stat-label")
                        yield Label("0", id="stat-processed", classes="stat-value")
                    with Vertical(classes="stat-card"):
                        yield Label("TIME", classes="stat-label")
                        yield Label("00:00", id="stat-time", classes="stat-value")
                    with Vertical(classes="stat-card"):
                        yield Label("STATUS", classes="stat-label")
                        yield Label("Ready", id="stat-status", classes="stat-value")

                # Logs & Progress
                yield Label("System Logs", classes="title")
                yield ProgressBar(total=100, show_eta=False, id="progress-bar")
                yield Log(id="system-log")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("ID", "Time", "Original", "Processed", "Action")
        self.log_message("INFO", "SubtitleLab TUI initialized.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "btn-import":
            self.action_import_file()
        elif btn_id == "btn-start":
            self.action_start_processing()
        elif btn_id == "btn-cancel":
            self.action_cancel_processing()
        elif btn_id == "btn-export":
            self.action_export_file()
        elif btn_id == "btn-settings":
            self.action_open_settings()

    def action_open_settings(self) -> None:
        self.app.push_screen(SettingsScreen(self.app.config))

    def action_import_file(self) -> None:
        self.log_message(
            "INFO", "File picker not implemented yet. Please drag & drop or implement FilePicker."
        )
        self.log_message(
            "WARNING", "Please place .srt file in current directory and update code to load it."
        )

    def log_message(self, level: str, message: str) -> None:
        log = self.query_one(Log)
        timestamp = datetime.now().strftime("%H:%M:%S")
        log.write_line(f"[{timestamp}] [{level}] {message}")

    @work(exclusive=True)
    async def action_start_processing(self) -> None:
        if not self.current_file:
            self.log_message("ERROR", "No file selected.")
            return

        self.query_one("#btn-start", Button).disabled = True
        self.query_one("#btn-cancel", Button).disabled = False
        self.query_one("#stat-status", Label).update("Processing...")

        self.processor = SubtitleProcessor(self.app.config)

        self.processor.callbacks = ProcessorCallbacks(
            on_progress=self.update_progress,
            on_batch_complete=self.update_batch,
            on_log=self.log_message_thread,
            on_error=self.log_error_thread,
        )
        # TODO: Implement proper FilePicker
        # For now, let's assume a test file exists or user provides path
        self.log_message(
            "WARNING", "Please place .srt file in current directory and update code to load it."
        )

    def log_message(self, level: str, message: str) -> None:
        log = self.query_one(Log)
        timestamp = datetime.now().strftime("%H:%M:%S")
        log.write_line(f"[{timestamp}] [{level}] {message}")

    @work(exclusive=True)
    async def action_start_processing(self) -> None:
        if not self.current_file:
            self.log_message("ERROR", "No file selected.")
            return

        self.query_one("#btn-start", Button).disabled = True
        self.query_one("#btn-cancel", Button).disabled = False
        self.query_one("#stat-status", Label).update("Processing...")

        self.processor = SubtitleProcessor(self.app.config)

        # Setup callbacks
        # Note: These run in the worker thread/task, so we must schedule UI updates
        self.processor.callbacks = ProcessorCallbacks(
            on_progress=self.update_progress,
            on_batch_complete=self.update_batch,
            on_log=self.log_message_thread,
            on_error=self.log_error_thread,
        )

        try:
            self.start_time = datetime.now()
            await self.processor.process()
            self.log_message("INFO", "Processing completed successfully.")
            self.query_one("#stat-status", Label).update("Completed")
            self.query_one("#btn-export", Button).disabled = False
        except Exception as e:
            self.log_message("ERROR", f"Processing failed: {e}")
            self.query_one("#stat-status", Label).update("Error")
        finally:
            self.query_one("#btn-start", Button).disabled = False
            self.query_one("#btn-cancel", Button).disabled = True

    def update_progress(self, progress: float, message: str) -> None:
        self.app.call_from_thread(self._update_progress_ui, progress)

    def _update_progress_ui(self, progress: float) -> None:
        bar = self.query_one(ProgressBar)
        bar.progress = progress * 100

    def update_batch(self, result: BatchResult) -> None:
        self.app.call_from_thread(self._update_batch_ui, result)

    def _update_batch_ui(self, result: BatchResult) -> None:
        table = self.query_one(DataTable)
        for entry in result.entries:
            time_str = f"{entry.start:.2f} -> {entry.end:.2f}"

            processed_text = entry.text
            if entry.action.value == "delete":
                processed_text = "[red][deleted][/red]"
            elif entry.action.value == "merge":
                processed_text = f"[yellow]{entry.text}[/yellow]"
            elif entry.original_text != entry.text:
                processed_text = f"[green]{entry.text}[/green]"

            table.add_row(
                str(entry.id), time_str, entry.original_text, processed_text, entry.action.value
            )

        processed_count = table.row_count
        self.query_one("#stat-processed", Label).update(str(processed_count))

    def log_message_thread(self, level: str, message: str) -> None:
        self.app.call_from_thread(self.log_message, level, message)

    def log_error_thread(self, error: str, context: str) -> None:
        self.app.call_from_thread(self.log_message, "ERROR", f"{context}: {error}")

    def action_cancel_processing(self) -> None:
        if self.processor:
            self.processor.cancel()
            self.log_message("WARNING", "Processing cancelled.")

    def action_export_file(self) -> None:
        pass


class SubtitleLabApp(App):
    CSS_PATH = "styles.tcss"
    TITLE = "SubtitleLab TUI"
    SUB_TITLE = "Intelligent Subtitle Fixer"

    def __init__(self):
        super().__init__()
        self.config = AppConfig.load()

    def on_mount(self) -> None:
        self.push_screen(Dashboard())


def run_app():
    app = SubtitleLabApp()
    app.run()
