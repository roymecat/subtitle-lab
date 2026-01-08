import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import flet as ft

from .theme import Theme, ThemeMode, get_theme
from .components import (
    GlassCard,
    AnimatedProgressRing,
    SubtitleListItem,
    ProcessedResultCard,
    LogEntry,
    StatsCard,
)
from .settings_dialog import SettingsDialog
from ..core.processor import SubtitleProcessor, ProcessorCallbacks
from ..core.config import AppConfig
from ..core.models import BatchResult, ProcessingStats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubtitleLabApp:
    def __init__(self):
        self.config = AppConfig.load()
        self.theme_mode = ThemeMode.DARK if self.config.ui.theme_mode == "dark" else ThemeMode.LIGHT
        self.theme: Theme = get_theme(self.theme_mode)

        self.current_file: Optional[Path] = None
        self.is_processing: bool = False
        self.processor: Optional[SubtitleProcessor] = None

        self.page: Optional[ft.Page] = None
        self.original_list: Optional[ft.ListView] = None
        self.processed_list: Optional[ft.ListView] = None
        self.log_list: Optional[ft.ListView] = None
        self.progress_ring: Optional[AnimatedProgressRing] = None
        self.status_text: Optional[ft.Text] = None
        self.start_btn: Optional[ft.ElevatedButton] = None
        self.cancel_btn: Optional[ft.ElevatedButton] = None
        self.export_btn: Optional[ft.ElevatedButton] = None
        self.file_picker: Optional[ft.FilePicker] = None

        self.stat_total: Optional[StatsCard] = None
        self.stat_processed: Optional[StatsCard] = None
        self.stat_time: Optional[StatsCard] = None

    async def main(self, page: ft.Page):
        self.page = page
        self.page.title = "SubtitleLab"
        self.page.theme_mode = (
            ft.ThemeMode.DARK if self.theme_mode == ThemeMode.DARK else ft.ThemeMode.LIGHT
        )
        self.page.bgcolor = self.theme.background
        self.page.padding = 20
        self.page.fonts = {
            "JetBrains Mono": "https://github.com/JetBrains/JetBrainsMono/raw/master/fonts/ttf/JetBrainsMono-Regular.ttf",
        }

        self.file_picker = ft.FilePicker()
        self.page.overlay.append(self.file_picker)

        header = self._build_header()
        content = self._build_main_content()
        bottom_panel = self._build_bottom_panel()

        self.page.add(
            ft.Column(
                controls=[
                    header,
                    ft.Divider(height=1, color=self.theme.border),
                    content,
                    ft.Divider(height=1, color=self.theme.border),
                    bottom_panel,
                ],
                expand=True,
                spacing=0,
            )
        )

        self._log("INFO", "SubtitleLab initialized. Ready to import.")

    def _build_header(self) -> ft.Control:
        return ft.Container(
            content=ft.Row(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Icon(
                                name="movie_creation_outlined", color=self.theme.primary, size=32
                            ),
                            ft.Text(
                                "SubtitleLab",
                                size=24,
                                weight=ft.FontWeight.BOLD,
                                color=self.theme.text_primary,
                            ),
                            ft.Container(
                                content=ft.Text(
                                    "BETA",
                                    size=10,
                                    color=self.theme.background,
                                    weight=ft.FontWeight.BOLD,
                                ),
                                bgcolor=self.theme.accent,
                                padding=ft.padding.symmetric(horizontal=6, vertical=2),
                                border_radius=4,
                            ),
                        ],
                        spacing=12,
                    ),
                    ft.Row(
                        controls=[
                            ft.IconButton(
                                icon="dark_mode"
                                if self.theme_mode == ThemeMode.LIGHT
                                else "light_mode",
                                icon_color=self.theme.text_secondary,
                                tooltip="Toggle Theme",
                                on_click=self._toggle_theme,
                            ),
                            ft.IconButton(
                                icon="settings_outlined",
                                icon_color=self.theme.text_secondary,
                                tooltip="Settings",
                                on_click=self._on_settings_click,
                            ),
                            ft.Container(
                                content=ft.Text(
                                    "John Doe",
                                    color=self.theme.text_primary,
                                    weight=ft.FontWeight.BOLD,
                                ),
                                padding=ft.padding.symmetric(horizontal=12, vertical=8),
                                border_radius=self.theme.radius.MD,
                                bgcolor=self.theme.surface_light,
                            ),
                        ],
                        spacing=8,
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            padding=ft.padding.symmetric(vertical=10),
        )

    def _build_main_content(self) -> ft.Control:
        self.original_list = ft.ListView(
            expand=True,
            spacing=10,
            padding=10,
        )

        left_panel = ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Text(
                            "ORIGINAL SUBTITLES",
                            color=self.theme.text_secondary,
                            size=12,
                            weight=ft.FontWeight.BOLD,
                        ),
                        ft.Container(
                            content=ft.Row(
                                [
                                    ft.Icon(name="upload_file", size=16, color=self.theme.primary),
                                    ft.Text(
                                        "Import File",
                                        color=self.theme.primary,
                                        size=12,
                                        weight=ft.FontWeight.BOLD,
                                    ),
                                ],
                                spacing=4,
                            ),
                            on_click=self._on_import_click,
                            padding=ft.padding.symmetric(horizontal=12, vertical=8),
                            border_radius=self.theme.radius.SM,
                            bgcolor=ft.Colors.with_opacity(0.1, self.theme.primary),
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                GlassCard(
                    theme=self.theme,
                    content=self.original_list,
                    expand=True,
                    padding=0,
                ),
            ],
            expand=True,
        )

        self.processed_list = ft.ListView(
            expand=True,
            spacing=10,
            padding=10,
            auto_scroll=True,
        )

        right_panel = ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Text(
                            "PROCESSED OUTPUT",
                            color=self.theme.text_secondary,
                            size=12,
                            weight=ft.FontWeight.BOLD,
                        ),
                        ft.Row(
                            controls=[
                                self._build_action_button(
                                    "Start Processing",
                                    "play_arrow",
                                    self._on_start_click,
                                    "primary",
                                ),
                                self._build_action_button(
                                    "Cancel",
                                    "stop",
                                    self._on_cancel_click,
                                    "error",
                                    disabled=True,
                                ),
                                self._build_action_button(
                                    "Export",
                                    "download",
                                    self._on_export_click,
                                    "success",
                                    disabled=True,
                                ),
                            ],
                            spacing=8,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                GlassCard(
                    theme=self.theme,
                    content=self.processed_list,
                    expand=True,
                    padding=0,
                ),
            ],
            expand=True,
        )

        return ft.Row(
            controls=[left_panel, right_panel],
            expand=True,
            spacing=20,
        )

    def _build_action_button(
        self, text: str, icon: str, on_click, variant: str, disabled: bool = False
    ):
        style = self.theme.get_button_style(variant)
        btn = ft.ElevatedButton(
            text=text,
            icon=icon,
            style=ft.ButtonStyle(
                color=style["color"],
                bgcolor={
                    ft.ControlState.DEFAULT: style["bgcolor"],
                    ft.ControlState.DISABLED: self.theme.surface_light,
                },
                shape=ft.RoundedRectangleBorder(radius=style["border_radius"]),
                padding=style["padding"],
            ),
            on_click=on_click,
            disabled=disabled,
        )
        if text == "Start Processing":
            self.start_btn = btn
        elif text == "Cancel":
            self.cancel_btn = btn
        elif text == "Export":
            self.export_btn = btn
        return btn

    def _build_bottom_panel(self) -> ft.Control:
        self.progress_ring = AnimatedProgressRing(theme=self.theme, size=60, stroke_width=6)

        self.stat_total = StatsCard(self.theme, "list", "Total Lines", "0")
        self.stat_processed = StatsCard(self.theme, "check_circle", "Processed", "0", "0%", True)
        self.stat_time = StatsCard(self.theme, "timer", "Time Elapsed", "00:00")

        self.log_list = ft.ListView(
            expand=True,
            spacing=4,
            padding=10,
            auto_scroll=True,
            height=100,
        )

        log_panel = ft.Container(
            content=self.log_list,
            bgcolor=self.theme.surface,
            border_radius=self.theme.radius.MD,
            border=ft.border.all(1, self.theme.border),
            expand=True,
        )

        return ft.Container(
            content=ft.Row(
                controls=[
                    self.progress_ring,
                    ft.VerticalDivider(width=1, color=self.theme.border),
                    self.stat_total,
                    self.stat_processed,
                    self.stat_time,
                    ft.VerticalDivider(width=1, color=self.theme.border),
                    ft.Column(
                        controls=[
                            ft.Text(
                                "SYSTEM LOGS",
                                color=self.theme.text_secondary,
                                size=10,
                                weight=ft.FontWeight.BOLD,
                            ),
                            log_panel,
                        ],
                        expand=True,
                        spacing=4,
                    ),
                ],
                spacing=20,
                height=120,
            ),
            padding=ft.padding.only(top=10),
        )

    def _toggle_theme(self, e):
        self.theme_mode = ThemeMode.LIGHT if self.theme_mode == ThemeMode.DARK else ThemeMode.DARK
        self.theme = get_theme(self.theme_mode)
        self.page.theme_mode = (
            ft.ThemeMode.LIGHT if self.theme_mode == ThemeMode.LIGHT else ft.ThemeMode.DARK
        )
        self.page.bgcolor = self.theme.background
        self.page.update()
        self._log(
            "INFO",
            f"Theme switched to {self.theme_mode.name}. (Restart required for full effect on custom components)",
        )

    def _on_settings_click(self, e):
        def on_save(updated_config: AppConfig):
            self.config = updated_config
            self.config.save()
            self._log("INFO", "Settings saved successfully")

        dialog = SettingsDialog(self.config, self.theme, on_save)
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    def _log(self, level: str, message: str):
        if not self.log_list:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_list.controls.append(LogEntry(self.theme, timestamp, level, message))
        if self.page:
            self.log_list.update()

    async def _on_import_click(self, e):
        files = await self.file_picker.pick_files(
            allow_multiple=False, allowed_extensions=["srt", "ass", "ssa"]
        )
        if not files:
            return

        file_path = files[0].path
        self.current_file = Path(file_path)
        self._log("INFO", f"Selected file: {self.current_file}")
        await self._load_subtitles_async(self.current_file)

    async def _load_subtitles_async(self, path: Path):
        self.processor = SubtitleProcessor(self.config)
        self._log("INFO", "Loading subtitles...")
        self.start_btn.disabled = True
        self.start_btn.update()

        try:
            loop = asyncio.get_running_loop()
            entries = await loop.run_in_executor(None, self.processor.load_subtitles, path)

            self.original_list.controls.clear()
            self.processed_list.controls.clear()

            total_entries = len(entries)
            CHUNK_SIZE = 50

            if total_entries <= CHUNK_SIZE:
                self._add_entries_to_list(entries)
                self.original_list.update()
            else:
                self._log("INFO", f"Large file ({total_entries} lines), loading in chunks...")
                await self._load_entries_chunked(entries, CHUNK_SIZE, total_entries)

            self.stat_total.content.controls[1].controls[1].value = str(total_entries)
            self.stat_total.update()
            self.processed_list.update()

            self.start_btn.disabled = False
            self.start_btn.update()
            self._log("INFO", f"Loaded {total_entries} subtitles.")

        except Exception as ex:
            self._log("ERROR", f"Failed to load file: {str(ex)}")
            self.start_btn.disabled = True
            self.start_btn.update()

    def _add_entries_to_list(self, entries):
        for entry in entries:
            self.original_list.controls.append(
                SubtitleListItem(
                    self.theme,
                    entry.id,
                    f"{entry.start:.2f}",
                    f"{entry.end:.2f}",
                    entry.text,
                    "pending",
                )
            )

    async def _load_entries_chunked(self, entries, chunk_size: int, total: int):
        for i in range(0, total, chunk_size):
            chunk = entries[i : i + chunk_size]
            self._add_entries_to_list(chunk)
            self.original_list.update()

            loaded = min(i + chunk_size, total)
            self.stat_total.content.controls[1].controls[1].value = f"{loaded}/{total}"
            self.stat_total.update()

            await asyncio.sleep(0)

    async def _on_start_click(self, e):
        if not self.processor or not self.current_file:
            return

        self.is_processing = True
        self.start_btn.disabled = True
        self.cancel_btn.disabled = False
        self.export_btn.disabled = True
        self.start_btn.update()
        self.cancel_btn.update()
        self.export_btn.update()

        self.processor.callbacks = ProcessorCallbacks(
            on_progress=self._on_progress_update,
            on_batch_complete=self._on_batch_complete,
            on_log=self._on_processor_log,
            on_error=self._on_processor_error,
        )

        try:
            stats = await self.processor.process()
            self._log("INFO", "Processing finished successfully.")
            self.export_btn.disabled = False
            self.export_btn.update()
        except Exception as ex:
            self._log("ERROR", f"Processing failed: {str(ex)}")
        finally:
            self.is_processing = False
            self.start_btn.disabled = False
            self.cancel_btn.disabled = True
            self.start_btn.update()
            self.cancel_btn.update()
            await self._cleanup_processor()

    async def _on_cancel_click(self, e):
        if not self.processor or not self.is_processing:
            return

        self._log("WARNING", "Cancelling processing...")
        self.processor.cancel()

        self.cancel_btn.disabled = True
        self.cancel_btn.update()

    async def _cleanup_processor(self):
        if self.processor:
            try:
                await self.processor.close()
                self._log("INFO", "Processor resources cleaned up.")
            except Exception as ex:
                self._log("WARNING", f"Error during cleanup: {str(ex)}")

    async def _on_export_click(self, e):
        if not self.processor:
            return

        save_path = await self.file_picker.save_file(
            file_name=f"{self.current_file.stem}_processed.srt", allowed_extensions=["srt"]
        )
        if not save_path:
            return

        try:
            self.processor.save_results(save_path)
            self._log("INFO", f"Saved results to {save_path}")

            snack = ft.SnackBar(
                ft.Text(f"Successfully saved to {save_path}"), bgcolor=self.theme.success
            )
            self.page.overlay.append(snack)
            snack.open = True
            self.page.update()

        except Exception as ex:
            self._log("ERROR", f"Failed to save: {str(ex)}")

    def _on_progress_update(self, progress: float, message: str):
        self.progress_ring.progress_ring.value = progress
        self.progress_ring.percentage_text.value = f"{int(progress * 100)}%"
        self.progress_ring.update()

        self._update_elapsed_time()

    def _on_batch_complete(self, result: BatchResult):
        for entry in result.entries:
            self.processed_list.controls.append(
                ProcessedResultCard(
                    self.theme,
                    entry.original_text,
                    entry.text,
                    entry.action.value,
                    entry.reason,
                    f"{entry.start:.2f}",
                    f"{entry.end:.2f}",
                )
            )
        self.processed_list.update()

        processed_count = len(self.processed_list.controls)
        total_count = len(self.original_list.controls) if self.original_list else 0
        percentage = (processed_count / total_count * 100) if total_count > 0 else 0

        self.stat_processed.content.controls[1].controls[1].value = str(processed_count)
        if (
            hasattr(self.stat_processed.content.controls[1], "controls")
            and len(self.stat_processed.content.controls[1].controls) > 2
        ):
            self.stat_processed.content.controls[1].controls[2].value = f"{percentage:.0f}%"
        self.stat_processed.update()

        self._update_elapsed_time()
        self._log_batch_stats(result)

    def _update_elapsed_time(self):
        if not self.processor or not hasattr(self.processor, "_state"):
            return

        stats = self.processor._state.stats
        if stats and stats.start_time:
            elapsed = (datetime.now() - stats.start_time).total_seconds()
            minutes, seconds = divmod(int(elapsed), 60)
            self.stat_time.content.controls[1].controls[1].value = f"{minutes:02d}:{seconds:02d}"
            self.stat_time.update()

    def _log_batch_stats(self, result: BatchResult):
        if not result.success:
            return

        action_counts = {}
        for entry in result.entries:
            action = entry.action.value
            action_counts[action] = action_counts.get(action, 0) + 1

        stats_parts = [f"{k}:{v}" for k, v in sorted(action_counts.items())]
        stats_str = ", ".join(stats_parts) if stats_parts else "empty"

        self._log(
            "INFO",
            f"Batch {result.batch_index + 1}: {len(result.entries)} entries ({stats_str})",
        )

    def _on_processor_log(self, level: str, message: str):
        self._log(level, message)

    def _on_processor_error(self, error: Exception, context: str):
        self._log("ERROR", f"{context}: {str(error)}")


if __name__ == "__main__":
    app = SubtitleLabApp()
    ft.app(target=app.main)
