from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QTextEdit,
    QSizePolicy,
    QScrollArea,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from .theme import Theme
from .translations import get_translation


def _tr(text: str, lang: str = "zh_CN") -> str:
    """Module-level translation helper."""
    return get_translation(text, lang)


class SubtitleListItem(QFrame):
    def __init__(self, sub_id: int, start_time: str, end_time: str, text: str, theme: Theme):
        super().__init__()
        self.theme = theme
        self._setup_ui(sub_id, start_time, end_time, text)

    def _setup_ui(self, sub_id: int, start_time: str, end_time: str, text: str):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {self.theme.surface};
                border: 1px solid {self.theme.border};
                border-radius: {self.theme.radius.MD}px;
                padding: 8px;
            }}
            QFrame:hover {{
                background-color: {self.theme.surface_light};
                border-color: {self.theme.primary};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        id_label = QLabel(f"#{sub_id}")
        id_label.setStyleSheet(
            f"color: {self.theme.text_secondary}; font-weight: bold; font-size: 12px;"
        )
        id_label.setFixedWidth(40)
        layout.addWidget(id_label)

        time_label = QLabel(f"{start_time} → {end_time}")
        time_label.setStyleSheet(f"""
            background-color: {self.theme.surface_light};
            color: {self.theme.accent};
            padding: 4px 8px;
            border-radius: {self.theme.radius.SM}px;
            font-family: monospace;
            font-size: 12px;
        """)
        layout.addWidget(time_label)

        text_label = QLabel(text)
        text_label.setStyleSheet(f"color: {self.theme.text_primary}; font-size: 14px;")
        text_label.setWordWrap(True)
        layout.addWidget(text_label, 1)


class ProcessedResultCard(QFrame):
    def __init__(
        self,
        original_text: str,
        processed_text: str,
        action: str,
        reason: str,
        start_time: str,
        end_time: str,
        theme: Theme,
    ):
        super().__init__()
        self.theme = theme
        self._setup_ui(original_text, processed_text, action, reason, start_time, end_time)

    def _setup_ui(
        self,
        original_text: str,
        processed_text: str,
        action: str,
        reason: str,
        start_time: str,
        end_time: str,
    ):
        action_colors = {
            "merge": self.theme.secondary,
            "delete": self.theme.error,
            "correct": self.theme.success,
            "keep": self.theme.text_secondary,
        }
        action_icons = {
            "merge": "🔗",
            "delete": "🗑️",
            "correct": "✏️",
            "keep": "✓",
        }

        color = action_colors.get(action.lower(), self.theme.text_secondary)
        icon = action_icons.get(action.lower(), "•")

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {self.theme.surface};
                border: 1px solid {self.theme.border};
                border-radius: {self.theme.radius.LG}px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()

        action_label = QLabel(f"{icon} {action.upper()}")
        action_label.setStyleSheet(f"""
            background-color: {color}20;
            color: {color};
            padding: 4px 8px;
            border-radius: {self.theme.radius.SM}px;
            font-weight: bold;
            font-size: 12px;
        """)
        header.addWidget(action_label)

        header.addStretch()

        time_label = QLabel(f"{start_time} - {end_time}")
        time_label.setStyleSheet(f"color: {self.theme.text_secondary}; font-size: 12px;")
        header.addWidget(time_label)

        layout.addLayout(header)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"background-color: {self.theme.border};")
        line.setFixedHeight(1)
        layout.addWidget(line)

        content = QHBoxLayout()

        original_section = QVBoxLayout()
        orig_title = QLabel(_tr("ORIGINAL"))
        orig_title.setStyleSheet(
            f"color: {self.theme.text_secondary}; font-size: 10px; font-weight: bold;"
        )
        original_section.addWidget(orig_title)
        orig_text = QLabel(original_text)
        orig_text.setStyleSheet(f"color: {self.theme.text_secondary}; font-size: 14px;")
        orig_text.setWordWrap(True)
        original_section.addWidget(orig_text)
        content.addLayout(original_section, 1)

        arrow = QLabel("→")
        arrow.setStyleSheet(f"color: {self.theme.border}; font-size: 20px;")
        arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content.addWidget(arrow)

        processed_section = QVBoxLayout()
        proc_title = QLabel(_tr("PROCESSED"))
        proc_title.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold;")
        processed_section.addWidget(proc_title)
        proc_text = QLabel(processed_text)
        proc_text.setStyleSheet(
            f"color: {self.theme.text_primary}; font-size: 14px; font-weight: 500;"
        )
        proc_text.setWordWrap(True)
        processed_section.addWidget(proc_text)
        content.addLayout(processed_section, 1)

        layout.addLayout(content)

        if reason:
            reason_label = QLabel(reason)
            reason_label.setStyleSheet(
                f"color: {self.theme.text_secondary}; font-size: 12px; font-style: italic;"
            )
            reason_label.setWordWrap(True)
            layout.addWidget(reason_label)


class StatsCard(QFrame):
    def __init__(self, label: str, value: str, theme: Theme):
        super().__init__()
        self.theme = theme
        self._setup_ui(label, value)

    def _setup_ui(self, label: str, value: str):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {self.theme.surface};
                border: 1px solid {self.theme.border};
                border-radius: {self.theme.radius.MD}px;
                padding: 8px;
            }}
        """)
        self.setFixedWidth(120)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        self.label_widget = QLabel(label)
        self.label_widget.setStyleSheet(f"color: {self.theme.text_secondary}; font-size: 12px;")
        layout.addWidget(self.label_widget)

        self.value_widget = QLabel(value)
        self.value_widget.setStyleSheet(
            f"color: {self.theme.text_primary}; font-size: 20px; font-weight: bold;"
        )
        layout.addWidget(self.value_widget)

    def set_value(self, value: str):
        self.value_widget.setText(value)


class LogPanel(QFrame):
    def __init__(self, theme: Theme):
        super().__init__()
        self.theme = theme
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {self.theme.surface};
                border: 1px solid {self.theme.border};
                border-radius: {self.theme.radius.MD}px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: transparent;
                border: none;
                color: {self.theme.text_primary};
                font-family: monospace;
                font-size: 12px;
            }}
        """)
        layout.addWidget(self.log_text)

    def add_log(self, timestamp: str, level: str, message: str):
        level_icons = {
            "INFO": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "PROCESSING": "🔄",
        }
        level_colors = {
            "INFO": self.theme.success,
            "WARNING": self.theme.warning,
            "ERROR": self.theme.error,
            "PROCESSING": self.theme.primary,
        }

        icon = level_icons.get(level.upper(), "🔹")
        color = level_colors.get(level.upper(), self.theme.text_secondary)

        html = f'<span style="color: {self.theme.text_secondary}">{timestamp}</span> '
        html += f"{icon} "
        html += f'<span style="color: {color if level.upper() == "ERROR" else self.theme.text_primary}">{message}</span>'

        self.log_text.append(html)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear(self):
        self.log_text.clear()
