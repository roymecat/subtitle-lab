from dataclasses import dataclass
from enum import Enum


class ThemeMode(Enum):
    LIGHT = "light"
    DARK = "dark"


@dataclass
class BorderRadius:
    SM: int = 4
    MD: int = 8
    LG: int = 12
    XL: int = 16


@dataclass
class Theme:
    mode: ThemeMode
    primary: str
    secondary: str
    accent: str
    success: str
    warning: str
    error: str
    background: str
    surface: str
    surface_light: str
    text_primary: str
    text_secondary: str
    border: str
    radius: BorderRadius

    def get_stylesheet(self) -> str:
        return f"""
        QMainWindow, QDialog {{
            background-color: {self.background};
        }}
        
        QWidget {{
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
            font-size: 13px;
            color: {self.text_primary};
        }}
        
        QLabel {{
            color: {self.text_primary};
        }}
        
        QLabel[class="secondary"] {{
            color: {self.text_secondary};
            font-size: 12px;
        }}
        
        QLabel[class="title"] {{
            font-size: 24px;
            font-weight: bold;
        }}
        
        QLabel[class="subtitle"] {{
            font-size: 16px;
            font-weight: bold;
        }}
        
        QPushButton {{
            background-color: {self.primary};
            color: white;
            border: none;
            border-radius: {self.radius.MD}px;
            padding: 8px 16px;
            font-weight: bold;
        }}
        
        QPushButton:hover {{
            background-color: {self._lighten(self.primary, 10)};
        }}
        
        QPushButton:pressed {{
            background-color: {self._darken(self.primary, 10)};
        }}
        
        QPushButton:disabled {{
            background-color: {self.surface_light};
            color: {self.text_secondary};
        }}
        
        QPushButton[class="secondary"] {{
            background-color: {self.surface_light};
            color: {self.text_primary};
            border: 1px solid {self.border};
        }}
        
        QPushButton[class="secondary"]:hover {{
            background-color: {self.surface};
            border-color: {self.primary};
        }}
        
        QPushButton[class="success"] {{
            background-color: {self.success};
        }}
        
        QPushButton[class="success"]:hover {{
            background-color: {self._lighten(self.success, 10)};
        }}
        
        QPushButton[class="error"] {{
            background-color: {self.error};
        }}
        
        QPushButton[class="error"]:hover {{
            background-color: {self._lighten(self.error, 10)};
        }}
        
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
            background-color: {self.surface};
            border: 1px solid {self.border};
            border-radius: {self.radius.SM}px;
            padding: 8px;
            color: {self.text_primary};
        }}
        
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {self.primary};
        }}
        
        QComboBox {{
            background-color: {self.surface};
            border: 1px solid {self.border};
            border-radius: {self.radius.SM}px;
            padding: 8px;
            color: {self.text_primary};
        }}
        
        QComboBox:focus {{
            border-color: {self.primary};
        }}
        
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid {self.text_secondary};
            margin-right: 8px;
        }}
        
        QComboBox QAbstractItemView {{
            background-color: {self.surface};
            border: 1px solid {self.border};
            selection-background-color: {self.primary};
            selection-color: white;
        }}
        
        QTabWidget::pane {{
            border: 1px solid {self.border};
            border-radius: {self.radius.MD}px;
            background-color: {self.surface};
        }}
        
        QTabBar::tab {{
            background-color: {self.surface_light};
            color: {self.text_secondary};
            padding: 10px 20px;
            border-top-left-radius: {self.radius.MD}px;
            border-top-right-radius: {self.radius.MD}px;
            margin-right: 2px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {self.surface};
            color: {self.primary};
            font-weight: bold;
        }}
        
        QTabBar::tab:hover:!selected {{
            background-color: {self.surface};
        }}
        
        QScrollBar:vertical {{
            background-color: {self.surface};
            width: 10px;
            border-radius: 5px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {self.border};
            border-radius: 5px;
            min-height: 30px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {self.text_secondary};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        
        QScrollBar:horizontal {{
            background-color: {self.surface};
            height: 10px;
            border-radius: 5px;
        }}
        
        QScrollBar::handle:horizontal {{
            background-color: {self.border};
            border-radius: 5px;
            min-width: 30px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background-color: {self.text_secondary};
        }}
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
        
        QProgressBar {{
            background-color: {self.surface_light};
            border: none;
            border-radius: {self.radius.SM}px;
            text-align: center;
            color: {self.text_primary};
        }}
        
        QProgressBar::chunk {{
            background-color: {self.primary};
            border-radius: {self.radius.SM}px;
        }}
        
        QSlider::groove:horizontal {{
            background-color: {self.surface_light};
            height: 6px;
            border-radius: 3px;
        }}
        
        QSlider::handle:horizontal {{
            background-color: {self.primary};
            width: 16px;
            height: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }}
        
        QSlider::handle:horizontal:hover {{
            background-color: {self._lighten(self.primary, 10)};
        }}
        
        QSlider::sub-page:horizontal {{
            background-color: {self.primary};
            border-radius: 3px;
        }}
        
        QCheckBox {{
            spacing: 8px;
        }}
        
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border-radius: 4px;
            border: 2px solid {self.border};
            background-color: {self.surface};
        }}
        
        QCheckBox::indicator:checked {{
            background-color: {self.primary};
            border-color: {self.primary};
        }}
        
        QGroupBox {{
            font-weight: bold;
            border: 1px solid {self.border};
            border-radius: {self.radius.MD}px;
            margin-top: 12px;
            padding-top: 12px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 8px;
            color: {self.text_primary};
        }}
        
        QListWidget, QTreeWidget, QTableWidget {{
            background-color: {self.surface};
            border: 1px solid {self.border};
            border-radius: {self.radius.MD}px;
            padding: 4px;
            alternate-background-color: {self.surface_light};
        }}
        
        QListWidget::item, QTreeWidget::item {{
            padding: 8px;
            border-radius: {self.radius.SM}px;
        }}
        
        QListWidget::item:selected, QTreeWidget::item:selected {{
            background-color: {self.primary};
            color: white;
        }}
        
        QListWidget::item:hover:!selected, QTreeWidget::item:hover:!selected {{
            background-color: {self.surface_light};
        }}
        
        QHeaderView::section {{
            background-color: {self.surface_light};
            color: {self.text_primary};
            padding: 8px;
            border: none;
            border-bottom: 1px solid {self.border};
            font-weight: bold;
        }}
        
        QSplitter::handle {{
            background-color: {self.border};
        }}
        
        QSplitter::handle:horizontal {{
            width: 2px;
        }}
        
        QSplitter::handle:vertical {{
            height: 2px;
        }}
        
        QToolTip {{
            background-color: {self.surface};
            color: {self.text_primary};
            border: 1px solid {self.border};
            border-radius: {self.radius.SM}px;
            padding: 4px 8px;
        }}
        
        QMenu {{
            background-color: {self.surface};
            border: 1px solid {self.border};
            border-radius: {self.radius.MD}px;
            padding: 4px;
        }}
        
        QMenu::item {{
            padding: 8px 24px;
            border-radius: {self.radius.SM}px;
        }}
        
        QMenu::item:selected {{
            background-color: {self.primary};
            color: white;
        }}
        
        QStatusBar {{
            background-color: {self.surface};
            border-top: 1px solid {self.border};
        }}
        
        QMessageBox {{
            background-color: {self.surface};
        }}
        """

    def _lighten(self, color: str, percent: int) -> str:
        color = color.lstrip("#")
        r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
        r = min(255, r + int((255 - r) * percent / 100))
        g = min(255, g + int((255 - g) * percent / 100))
        b = min(255, b + int((255 - b) * percent / 100))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _darken(self, color: str, percent: int) -> str:
        color = color.lstrip("#")
        r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
        r = max(0, r - int(r * percent / 100))
        g = max(0, g - int(g * percent / 100))
        b = max(0, b - int(b * percent / 100))
        return f"#{r:02x}{g:02x}{b:02x}"


DARK_THEME = Theme(
    mode=ThemeMode.DARK,
    primary="#6366F1",
    secondary="#8B5CF6",
    accent="#06B6D4",
    success="#10B981",
    warning="#F59E0B",
    error="#EF4444",
    background="#0F172A",
    surface="#1E293B",
    surface_light="#334155",
    text_primary="#F8FAFC",
    text_secondary="#94A3B8",
    border="#475569",
    radius=BorderRadius(),
)

LIGHT_THEME = Theme(
    mode=ThemeMode.LIGHT,
    primary="#6366F1",
    secondary="#8B5CF6",
    accent="#06B6D4",
    success="#10B981",
    warning="#F59E0B",
    error="#EF4444",
    background="#F8FAFC",
    surface="#FFFFFF",
    surface_light="#F1F5F9",
    text_primary="#0F172A",
    text_secondary="#64748B",
    border="#E2E8F0",
    radius=BorderRadius(),
)


def get_theme(mode: ThemeMode) -> Theme:
    return DARK_THEME if mode == ThemeMode.DARK else LIGHT_THEME
