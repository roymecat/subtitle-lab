from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ContentSwitcher
from textual.widgets import (
    Header,
    Footer,
    Button,
    DataTable,
    Log,
    ProgressBar,
    Static,
    Label,
    ListView,
    ListItem,
    DirectoryTree,
)
from textual.screen import Screen
from textual.binding import Binding
from textual.reactive import reactive
from textual import work
from textual.command import Hit, Provider, DiscoveryHit

from pathlib import Path
import asyncio
from datetime import datetime

from ..core.config import AppConfig
from ..core.processor import SubtitleProcessor, ProcessorCallbacks
from ..core.models import BatchResult
from .screens.settings import SettingsScreen
from .screens.file_picker import FilePickerScreen


class SubtitleLabApp(App):
    CSS_PATH = "styles.tcss"
    TITLE = "SubtitleLab"
    SUB_TITLE = "Pro Subtitle Editor"

    BINDINGS = [
        Binding("ctrl+o", "import_file", "Open File"),
        Binding("ctrl+s", "start_processing", "Start"),
        Binding("ctrl+e", "export_file", "Export"),
        Binding("ctrl+comma", "open_settings", "Settings"),
        Binding("f1", "switch_tab('view-review')", "Review"),
        Binding("f2", "switch_tab('view-logs')", "Logs"),
        Binding("ctrl+p", "command_palette", "Commands"),
    ]

    current_file: reactive[Path | None] = reactive(None)
    processor: SubtitleProcessor | None = None
    start_time: datetime | None = None

    def __init__(self):
        super().__init__()
        self.config = AppConfig.load()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="app-grid"):
            # Sidebar
            with Vertical(id="sidebar"):
                yield Label("SubtitleLab", classes="brand")

                yield Label("MENU")
                yield ListView(
                    ListItem(Label("Review"), id="nav-review"),
                    ListItem(Label("Logs"), id="nav-logs"),
                    id="sidebar-nav",
                )

                yield Static(classes="spacer")

                with Vertical(classes="action-bar"):
                    yield Button("Import File", id="btn-import", variant="primary")
                    yield Button("Start", id="btn-start", variant="success", disabled=True)
                    yield Button("Export", id="btn-export", disabled=True)
                    yield Button("Settings", id="btn-settings")

            # Main Content
            with Vertical(id="content-area"):
                with ContentSwitcher(initial="view-review"):
                    # Review View
                    with Vertical(id="view-review"):
                        # Stats Header
                        with Horizontal(classes="stats-row"):
                            with Horizontal(classes="stat-item"):
                                yield Label("FILE: ")
                                yield Label("None", id="stat-filename", classes="stat-value")
                            with Horizontal(classes="stat-item"):
                                yield Label("TOTAL: ")
                                yield Label("0", id="stat-total", classes="stat-value")
                            with Horizontal(classes="stat-item"):
                                yield Label("PROCESSED: ")
                                yield Label("0", id="stat-processed", classes="stat-value")
                            with Horizontal(classes="stat-item"):
                                yield Label("STATUS: ")
                                yield Label("Idle", id="stat-status", classes="stat-value")

                        yield ProgressBar(total=100, show_eta=False, id="progress-bar")
                        yield DataTable(id="subtitle-table", cursor_type="row")

                    # Logs View
                    with Vertical(id="view-logs"):
                        yield Label("System Logs", classes="content-header")
                        yield Log(id="system-log")

        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("ID", "Time", "Original", "Processed", "Action")
        # Focus the list view initially
        self.query_one("#sidebar-nav").focus()
        self.log_message("INFO", "SubtitleLab TUI initialized.")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.id == "nav-review":
            self.query_one(ContentSwitcher).current = "view-review"
        elif event.item.id == "nav-logs":
            self.query_one(ContentSwitcher).current = "view-logs"

    def action_switch_tab(self, tab: str) -> None:
        self.query_one(ContentSwitcher).current = tab
        if tab == "view-review":
            self.query_one("#sidebar-nav").index = 0
        else:
            self.query_one("#sidebar-nav").index = 1

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "btn-import":
            self.action_import_file()
        elif btn_id == "btn-start":
            self.action_start_processing()
        elif btn_id == "btn-export":
            self.action_export_file()
        elif btn_id == "btn-settings":
            self.action_open_settings()

    def action_open_settings(self) -> None:
        self.push_screen(SettingsScreen(self.config))

    def action_import_file(self) -> None:
        def on_file_selected(path: Path | None):
            if path:
                self.current_file = path
                self.load_file(path)

        self.push_screen(FilePickerScreen(), on_file_selected)

    def load_file(self, path: Path) -> None:
        self.query_one("#stat-filename", Label).update(path.name)
        self.log_message("INFO", f"Loading file: {path}")

        # Reset UI
        table = self.query_one(DataTable)
        table.clear()

        try:
            # Pre-load subtitle count
            # Ideally this logic should be in core, but quick check here
            import pysubs2

            subs = pysubs2.load(str(path))
            self.query_one("#stat-total", Label).update(str(len(subs)))
            self.query_one("#btn-start", Button).disabled = False
            self.log_message("INFO", f"Loaded {len(subs)} subtitles.")
        except Exception as e:
            self.log_message("ERROR", f"Failed to load file: {e}")
            self.notify(f"Error loading file: {e}", severity="error")

    def log_message(self, level: str, message: str) -> None:
        log = self.query_one(Log)
        timestamp = datetime.now().strftime("%H:%M:%S")
        log.write_line(f"[{timestamp}] [{level}] {message}")

    @work(exclusive=True)
    async def action_start_processing(self) -> None:
        if not self.current_file:
            self.notify("No file selected", severity="error")
            return

        self.query_one("#btn-start", Button).disabled = True
        self.query_one("#btn-settings", Button).disabled = True
        self.query_one("#stat-status", Label).update("Processing...")

        self.processor = SubtitleProcessor(self.config)

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
            self.notify("Processing Complete!", severity="information")
        except Exception as e:
            self.log_message("ERROR", f"Processing failed: {e}")
            self.query_one("#stat-status", Label).update("Error")
            self.notify(f"Processing Failed: {e}", severity="error")
        finally:
            self.query_one("#btn-start", Button).disabled = False
            self.query_one("#btn-settings", Button).disabled = False

    def update_progress(self, progress: float, message: str) -> None:
        self.call_from_thread(self._update_progress_ui, progress)

    def _update_progress_ui(self, progress: float) -> None:
        bar = self.query_one(ProgressBar)
        bar.progress = progress * 100

    def update_batch(self, result: BatchResult) -> None:
        self.call_from_thread(self._update_batch_ui, result)

    def _update_batch_ui(self, result: BatchResult) -> None:
        table = self.query_one(DataTable)
        for entry in result.entries:
            time_str = f"{entry.start:.2f} -> {entry.end:.2f}"

            # Display ID or ID Range
            if len(entry.original_ids) > 1:
                id_str = f"{entry.original_ids[0]}-{entry.original_ids[-1]}"
            elif entry.original_ids:
                id_str = str(entry.original_ids[0])
            else:
                id_str = "-"

            processed_text = entry.text
            if entry.action.value == "delete":
                processed_text = "[red strike]deleted[/red strike]"
            elif entry.action.value == "merge":
                processed_text = f"[yellow]{entry.text}[/yellow]"
            elif entry.original_text != entry.text:
                processed_text = f"[green]{entry.text}[/green]"

            table.add_row(
                id_str, time_str, entry.original_text or "", processed_text, entry.action.value
            )

        processed_count = table.row_count
        self.query_one("#stat-processed", Label).update(str(processed_count))

    def log_message_thread(self, level: str, message: str) -> None:
        self.call_from_thread(self.log_message, level, message)

    def log_error_thread(self, error: Exception, context: str) -> None:
        self.call_from_thread(self.log_message, "ERROR", f"{context}: {error}")

    def action_export_file(self) -> None:
        if not self.processor or not self.current_file:
            return

        output_path = self.current_file.with_name(f"{self.current_file.stem}_processed.srt")
        try:
            self.processor.save_results(output_path)
            self.notify(f"Exported to {output_path.name}", severity="information")
            self.log_message("INFO", f"Exported to: {output_path}")
        except Exception as e:
            self.notify(f"Export Failed: {e}", severity="error")


def run_app():
    app = SubtitleLabApp()
    app.run()
