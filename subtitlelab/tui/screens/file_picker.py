from pathlib import Path
from typing import Iterable

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import DirectoryTree, Button, Label, Input
from textual.binding import Binding


class FilePickerScreen(ModalScreen[Path]):
    """Modal screen for selecting a file."""

    CSS = """
    FilePickerScreen {
        align: center middle;
        background: rgba(0,0,0,0.7);
    }

    #picker-container {
        width: 70%;
        height: 80%;
        background: $mantle;
        border: solid $lavender;
        padding: 1;
        layout: vertical;
    }

    #file-tree {
        height: 1fr;
        border: solid $surface;
        background: $base;
        margin: 1 0;
    }

    #path-display {
        background: $surface;
        border: none;
        margin-bottom: 1;
        color: $text;
    }

    .dialog-footer {
        height: 3;
        align: right middle;
    }

    .dialog-footer Button {
        width: 15;
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "select_current", "Select"),
    ]

    def __init__(self, initial_path: Path | None = None):
        super().__init__()
        self.initial_path = initial_path or Path.cwd()
        self.selected_path: Path | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-container"):
            yield Label("Select Subtitle File", classes="title")
            yield Input(str(self.initial_path), id="path-display", disabled=True)
            yield DirectoryTree(str(self.initial_path), id="file-tree")

            with Horizontal(classes="dialog-footer"):
                yield Button("Cancel", variant="default", id="btn-cancel")
                yield Button("Select", variant="primary", id="btn-select", disabled=True)

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self.selected_path = Path(event.path)
        self.query_one("#path-display", Input).value = str(self.selected_path)
        self.query_one("#btn-select", Button).disabled = False
        # Double click auto-selects
        self.dismiss(self.selected_path)

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        self.query_one("#path-display", Input).value = str(event.path)
        self.selected_path = None
        self.query_one("#btn-select", Button).disabled = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-select":
            if self.selected_path:
                self.dismiss(self.selected_path)
        elif event.button.id == "btn-cancel":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_select_current(self) -> None:
        # If tree has focus and a file is highlighted, select it
        tree = self.query_one(DirectoryTree)
        if tree.cursor_node and not tree.cursor_node.is_expanded and tree.cursor_node.data:
            # This is tricky without public API, sticking to button press for safety
            pass
