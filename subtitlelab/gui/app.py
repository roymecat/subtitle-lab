import sys
import asyncio
import locale
from pathlib import Path
from datetime import datetime
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QFrame,
    QScrollArea,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QFont, QColor

from .theme import Theme, ThemeMode, get_theme, DARK_THEME, LIGHT_THEME
from .settings_dialog import SettingsDialog
from .components import SubtitleListItem, ProcessedResultCard, StatsCard, LogPanel
from .translations import get_translation
from ..core.config import AppConfig
from ..core.processor import SubtitleProcessor, ProcessorCallbacks
from ..core.models import BatchResult


def detect_language() -> str:
    """Detect system language and return language code."""
    try:
        lang = locale.getdefaultlocale()[0]
        if lang and lang.startswith("zh"):
            return "zh_CN"
    except Exception:
        pass
    return "en_US"


# Global language setting
CURRENT_LANG = detect_language()


class ProcessorWorker(QThread):
    progress_updated = pyqtSignal(float, str)
    batch_completed = pyqtSignal(object)
    log_message = pyqtSignal(str, str)
    error_occurred = pyqtSignal(str, str)
    finished_processing = pyqtSignal(bool)

    def __init__(self, processor: SubtitleProcessor):
        super().__init__()
        self.processor = processor
        self._cancelled = False

    def run(self):
        try:
            self.processor.callbacks = ProcessorCallbacks(
                on_progress=lambda p, m: self.progress_updated.emit(p, m),
                on_batch_complete=lambda r: self.batch_completed.emit(r),
                on_log=lambda l, m: self.log_message.emit(l, m),
                on_error=lambda e, c: self.error_occurred.emit(str(e), c),
            )
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.processor.process())
            loop.close()
            self.finished_processing.emit(True)
        except Exception as e:
            self.error_occurred.emit(str(e), "Processing")
            self.finished_processing.emit(False)

    def cancel(self):
        self._cancelled = True
        if self.processor:
            self.processor.cancel()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = AppConfig.load()
        self.theme_mode = ThemeMode.DARK if self.config.ui.theme_mode == "dark" else ThemeMode.LIGHT
        self.theme = get_theme(self.theme_mode)
        self.lang = CURRENT_LANG

        self.current_file: Optional[Path] = None
        self.processor: Optional[SubtitleProcessor] = None
        self.worker: Optional[ProcessorWorker] = None
        self.start_time: Optional[datetime] = None

        self._init_ui()
        self._apply_theme()
        self._log("INFO", self._tr("SubtitleLab initialized. Ready to import."))

    def _tr(self, text: str) -> str:
        """Translate text using the current language."""
        return get_translation(text, self.lang)

    def _init_ui(self):
        self.setWindowTitle("SubtitleLab")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        main_layout.addWidget(self._create_header())
        main_layout.addWidget(self._create_separator())
        main_layout.addWidget(self._create_main_content(), 1)
        main_layout.addWidget(self._create_separator())
        main_layout.addWidget(self._create_bottom_panel())

    def _create_header(self) -> QWidget:
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)

        left_section = QHBoxLayout()

        title_label = QLabel("SubtitleLab")
        title_label.setProperty("class", "title")
        left_section.addWidget(title_label)

        beta_label = QLabel("BETA")
        beta_label.setStyleSheet(f"""
            background-color: {self.theme.accent};
            color: {self.theme.background};
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: bold;
        """)
        left_section.addWidget(beta_label)
        left_section.addStretch()

        right_section = QHBoxLayout()
        right_section.setSpacing(8)

        self.theme_btn = QPushButton("🌙" if self.theme_mode == ThemeMode.DARK else "☀️")
        self.theme_btn.setFixedSize(36, 36)
        self.theme_btn.setToolTip(self._tr("Toggle Theme"))
        self.theme_btn.clicked.connect(self._toggle_theme)
        self.theme_btn.setProperty("class", "secondary")
        right_section.addWidget(self.theme_btn)

        settings_btn = QPushButton("⚙️")
        settings_btn.setFixedSize(36, 36)
        settings_btn.setToolTip(self._tr("Settings"))
        settings_btn.clicked.connect(self._open_settings)
        settings_btn.setProperty("class", "secondary")
        right_section.addWidget(settings_btn)

        layout.addLayout(left_section)
        layout.addLayout(right_section)
        return header

    def _create_separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"background-color: {self.theme.border};")
        line.setFixedHeight(1)
        return line

    def _create_main_content(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setSizes([500, 500])
        return splitter

    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        header_layout = QHBoxLayout()
        title = QLabel(self._tr("ORIGINAL SUBTITLES"))
        title.setProperty("class", "secondary")
        header_layout.addWidget(title)
        header_layout.addStretch()

        import_btn = QPushButton("📁 " + self._tr("Import File"))
        import_btn.setProperty("class", "secondary")
        import_btn.clicked.connect(self._import_file)
        header_layout.addWidget(import_btn)
        layout.addLayout(header_layout)

        self.original_list = QListWidget()
        self.original_list.setAlternatingRowColors(True)
        layout.addWidget(self.original_list, 1)

        return panel

    def _create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        header_layout = QHBoxLayout()
        title = QLabel(self._tr("PROCESSED OUTPUT"))
        title.setProperty("class", "secondary")
        header_layout.addWidget(title)
        header_layout.addStretch()

        self.start_btn = QPushButton(self._tr("▶ Start"))
        self.start_btn.clicked.connect(self._start_processing)
        self.start_btn.setEnabled(False)
        header_layout.addWidget(self.start_btn)

        self.cancel_btn = QPushButton(self._tr("⏹ Cancel"))
        self.cancel_btn.setProperty("class", "error")
        self.cancel_btn.clicked.connect(self._cancel_processing)
        self.cancel_btn.setEnabled(False)
        header_layout.addWidget(self.cancel_btn)

        self.export_btn = QPushButton(self._tr("💾 Export"))
        self.export_btn.setProperty("class", "success")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        header_layout.addWidget(self.export_btn)

        layout.addLayout(header_layout)

        self.processed_list = QListWidget()
        self.processed_list.setAlternatingRowColors(True)
        layout.addWidget(self.processed_list, 1)

        return panel

    def _create_bottom_panel(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        progress_widget = QWidget()
        progress_layout = QVBoxLayout(progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(150)
        self.progress_bar.setFixedHeight(20)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("0%")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.progress_label)
        layout.addWidget(progress_widget)

        stats_widget = QWidget()
        stats_layout = QHBoxLayout(stats_widget)
        stats_layout.setSpacing(16)

        self.stat_total = StatsCard(self._tr("Total"), "0", self.theme)
        self.stat_processed = StatsCard(self._tr("Processed"), "0", self.theme)
        self.stat_time = StatsCard(self._tr("Time"), "00:00", self.theme)

        stats_layout.addWidget(self.stat_total)
        stats_layout.addWidget(self.stat_processed)
        stats_layout.addWidget(self.stat_time)
        layout.addWidget(stats_widget)

        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(4)

        log_title = QLabel(self._tr("SYSTEM LOGS"))
        log_title.setProperty("class", "secondary")
        log_layout.addWidget(log_title)

        self.log_panel = LogPanel(self.theme)
        self.log_panel.setFixedHeight(100)
        log_layout.addWidget(self.log_panel)

        layout.addWidget(log_widget, 1)

        return panel

    def _apply_theme(self):
        self.setStyleSheet(self.theme.get_stylesheet())

    def _toggle_theme(self):
        self.theme_mode = ThemeMode.LIGHT if self.theme_mode == ThemeMode.DARK else ThemeMode.DARK
        self.theme = get_theme(self.theme_mode)
        self.theme_btn.setText("🌙" if self.theme_mode == ThemeMode.DARK else "☀️")
        self._apply_theme()

        self.config.ui.theme_mode = self.theme_mode.value
        self.config.save()
        self._log("INFO", self._tr(f"Theme switched to {self.theme_mode.value}"))

    def _open_settings(self):
        dialog = SettingsDialog(self.config, self.theme, self)
        if dialog.exec():
            self.config = dialog.get_config()
            self.config.save()
            self._log("INFO", self._tr("Settings saved"))

    def _import_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self._tr("Import Subtitle File"),
            "",
            self._tr("Subtitle Files (*.srt *.ass *.ssa);;All Files (*)"),
        )
        if file_path:
            self.current_file = Path(file_path)
            self._load_subtitles()

    def _load_subtitles(self):
        if not self.current_file:
            return

        self._log("INFO", self._tr(f"Loading: {self.current_file.name}"))
        self.original_list.clear()
        self.processed_list.clear()

        try:
            self.processor = SubtitleProcessor(self.config)
            entries = self.processor.load_subtitles(self.current_file)

            for entry in entries:
                item = QListWidgetItem()
                widget = SubtitleListItem(
                    entry.id, f"{entry.start:.2f}", f"{entry.end:.2f}", entry.text, self.theme
                )
                item.setSizeHint(widget.sizeHint())
                self.original_list.addItem(item)
                self.original_list.setItemWidget(item, widget)

            self.stat_total.set_value(str(len(entries)))
            self.start_btn.setEnabled(True)
            self._log("INFO", self._tr(f"Loaded {len(entries)} subtitles"))

        except Exception as e:
            self._log("ERROR", str(e))
            QMessageBox.critical(self, self._tr("Error"), str(e))

    def _start_processing(self):
        if not self.processor:
            return

        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.processed_list.clear()
        self.start_time = datetime.now()

        self._start_timer()

        self.worker = ProcessorWorker(self.processor)
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.batch_completed.connect(self._on_batch_complete)
        self.worker.log_message.connect(self._log)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished_processing.connect(self._on_finished)
        self.worker.start()

    def _start_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_time)
        self.timer.start(1000)

    def _update_time(self):
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            minutes, seconds = divmod(int(elapsed), 60)
            self.stat_time.set_value(f"{minutes:02d}:{seconds:02d}")

    def _cancel_processing(self):
        if self.worker:
            self.worker.cancel()
            self._log("WARNING", self._tr("Processing cancelled"))

    def _on_progress(self, progress: float, message: str):
        percent = int(progress * 100)
        self.progress_bar.setValue(percent)
        self.progress_label.setText(f"{percent}%")

    def _on_batch_complete(self, result: BatchResult):
        for entry in result.entries:
            item = QListWidgetItem()
            widget = ProcessedResultCard(
                entry.original_text,
                entry.text,
                entry.action.value,
                entry.reason or "",
                f"{entry.start:.2f}",
                f"{entry.end:.2f}",
                self.theme,
            )
            item.setSizeHint(widget.sizeHint())
            self.processed_list.addItem(item)
            self.processed_list.setItemWidget(item, widget)

        self.stat_processed.set_value(str(self.processed_list.count()))

    def _on_error(self, error: str, context: str):
        self._log("ERROR", f"{context}: {error}")

    def _on_finished(self, success: bool):
        if hasattr(self, "timer"):
            self.timer.stop()

        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.export_btn.setEnabled(success)

        if success:
            self._log("INFO", self._tr("Processing completed successfully"))
        else:
            self._log("ERROR", self._tr("Processing failed"))

    def _export_results(self):
        if not self.processor or not self.current_file:
            return

        default_name = f"{self.current_file.stem}_processed.srt"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("Export Processed Subtitles"),
            default_name,
            self._tr("SRT Files (*.srt);;All Files (*)"),
        )

        if file_path:
            try:
                self.processor.save_results(file_path)
                self._log("INFO", self._tr(f"Exported to: {file_path}"))
                QMessageBox.information(
                    self, self._tr("Success"), self._tr(f"Successfully exported to:\n{file_path}")
                )
            except Exception as e:
                self._log("ERROR", str(e))
                QMessageBox.critical(self, self._tr("Error"), str(e))

    def _log(self, level: str, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_panel.add_log(timestamp, level, message)


def run_app():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
