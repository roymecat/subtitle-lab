from dataclasses import dataclass
from enum import Enum, auto


class ThemeMode(Enum):
    DARK = auto()
    LIGHT = auto()


@dataclass(frozen=True)
class BorderRadius:
    XL: int = 20
    LG: int = 16
    MD: int = 12
    SM: int = 8


@dataclass(frozen=True)
class Theme:
    mode: ThemeMode

    # Brand colors
    primary: str = "#6366F1"
    secondary: str = "#8B5CF6"
    accent: str = "#22D3EE"
    success: str = "#10B981"
    warning: str = "#F59E0B"
    error: str = "#EF4444"

    # Mode-specific colors
    background: str = ""
    surface: str = ""
    surface_light: str = ""
    text_primary: str = ""
    text_secondary: str = ""
    border: str = ""

    # Border radius
    radius: BorderRadius = None

    def __post_init__(self):
        object.__setattr__(self, "radius", BorderRadius())

    def get_container_style(self) -> dict:
        return {
            "bgcolor": self.surface,
            "border_radius": self.radius.MD,
            "border": f"1px solid {self.border}",
            "padding": 16,
        }

    def get_button_style(self, variant: str = "primary") -> dict:
        color_map = {
            "primary": self.primary,
            "secondary": self.secondary,
            "accent": self.accent,
            "success": self.success,
            "warning": self.warning,
            "error": self.error,
        }
        bg_color = color_map.get(variant, self.primary)

        return {
            "bgcolor": bg_color,
            "color": "#FFFFFF",
            "border_radius": self.radius.SM,
            "padding": 12,
        }

    def get_card_style(self) -> dict:
        return {
            "bgcolor": self.surface,
            "border_radius": self.radius.LG,
            "border": f"1px solid {self.border}",
            "padding": 20,
            "shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
        }


# Dark theme colors
_DARK_COLORS = {
    "background": "#0F0F1A",
    "surface": "#1A1A2E",
    "surface_light": "#252542",
    "text_primary": "#F8FAFC",
    "text_secondary": "#94A3B8",
    "border": "#334155",
}

# Light theme colors
_LIGHT_COLORS = {
    "background": "#F8FAFC",
    "surface": "#FFFFFF",
    "surface_light": "#F1F5F9",
    "text_primary": "#1E293B",
    "text_secondary": "#64748B",
    "border": "#E2E8F0",
}


def get_theme(mode: ThemeMode) -> Theme:
    colors = _DARK_COLORS if mode == ThemeMode.DARK else _LIGHT_COLORS
    return Theme(mode=mode, **colors)
