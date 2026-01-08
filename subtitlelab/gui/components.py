import flet as ft
from .theme import Theme
import asyncio


class GlassCard(ft.Container):
    def __init__(
        self,
        theme: Theme,
        content: ft.Control,
        width: int | None = None,
        height: int | None = None,
        padding: int = 20,
        margin: int = 0,
        expand: bool = False,
    ):
        super().__init__(
            content=content,
            width=width,
            height=height,
            padding=padding,
            margin=margin,
            border_radius=theme.radius.LG,
            border=ft.border.all(1, theme.border),
            bgcolor=ft.Colors.with_opacity(0.7, theme.surface),
            blur=ft.Blur(10, 10, ft.BlurTileMode.CLAMP),
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=10,
                color=ft.Colors.with_opacity(0.1, "#000000"),
                offset=ft.Offset(0, 4),
            ),
            animate=ft.animation.Animation(300, ft.AnimationCurve.EASE_OUT),
            expand=expand,
        )


class AnimatedProgressRing(ft.Stack):
    def __init__(
        self,
        theme: Theme,
        value: float = 0.0,
        size: int = 100,
        stroke_width: int = 8,
        color: str | None = None,
    ):
        self.theme = theme
        self.progress_ring = ft.ProgressRing(
            value=value,
            stroke_width=stroke_width,
            width=size,
            height=size,
            color=color or theme.primary,
            bgcolor=theme.surface_light,
        )
        self.percentage_text = ft.Text(
            value=f"{int(value * 100)}%",
            size=size * 0.25,
            weight=ft.FontWeight.BOLD,
            color=theme.text_primary,
            text_align=ft.TextAlign.CENTER,
        )

        super().__init__(
            controls=[
                self.progress_ring,
                ft.Container(
                    content=self.percentage_text,
                    alignment=ft.alignment.center,
                    width=size,
                    height=size,
                ),
            ],
            width=size,
            height=size,
        )

    async def update_progress(self, value: float):
        self.progress_ring.value = value
        self.percentage_text.value = f"{int(value * 100)}%"
        self.update()


class SubtitleListItem(ft.Container):
    def __init__(
        self,
        theme: Theme,
        sub_id: int,
        start_time: str,
        end_time: str,
        text: str,
        status: str = "pending",  # pending, processing, completed, error
        on_click=None,
    ):
        self.theme = theme

        status_colors = {
            "pending": theme.text_secondary,
            "processing": theme.warning,
            "completed": theme.success,
            "error": theme.error,
        }

        super().__init__(
            content=ft.Row(
                controls=[
                    # Status Indicator
                    ft.Container(
                        width=8,
                        height=8,
                        border_radius=4,
                        bgcolor=status_colors.get(status, theme.text_secondary),
                    ),
                    # ID
                    ft.Text(
                        f"#{sub_id}",
                        color=theme.text_secondary,
                        size=12,
                        weight=ft.FontWeight.BOLD,
                        width=40,
                    ),
                    # Timestamp
                    ft.Container(
                        content=ft.Text(
                            f"{start_time} → {end_time}",
                            color=theme.accent,
                            size=12,
                            font_family="monospace",
                        ),
                        bgcolor=ft.Colors.with_opacity(0.1, theme.accent),
                        padding=ft.padding.symmetric(horizontal=8, vertical=4),
                        border_radius=theme.radius.SM,
                    ),
                    # Text
                    ft.Text(
                        text,
                        color=theme.text_primary,
                        size=14,
                        overflow=ft.TextOverflow.ELLIPSIS,
                        expand=True,
                    ),
                ],
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=12,
            ),
            padding=ft.padding.all(12),
            border_radius=theme.radius.MD,
            bgcolor=theme.surface,
            border=ft.border.all(1, "transparent"),
            on_click=on_click,
            animate=ft.animation.Animation(200, ft.AnimationCurve.EASE_OUT),
            on_hover=self._on_hover,
        )

    def _on_hover(self, e):
        if e.data == "true":
            self.bgcolor = self.theme.surface_light
            self.border = ft.border.all(1, self.theme.border)
        else:
            self.bgcolor = self.theme.surface
            self.border = ft.border.all(1, "transparent")
        self.update()


class ProcessedResultCard(ft.Container):
    def __init__(
        self,
        theme: Theme,
        original_text: str,
        processed_text: str,
        action: str,  # merge, delete, correct, keep
        reason: str,
        start_time: str,
        end_time: str,
    ):
        self.theme = theme

        action_configs = {
            "merge": {"color": theme.secondary, "icon": "merge_type"},
            "delete": {"color": theme.error, "icon": "delete_outline"},
            "correct": {"color": theme.success, "icon": "auto_fix_high"},
            "keep": {"color": theme.text_secondary, "icon": "check_circle_outline"},
        }

        config = action_configs.get(action.lower(), action_configs["keep"])

        super().__init__(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Container(
                                content=ft.Row(
                                    controls=[
                                        ft.Icon(config["icon"], size=16, color=config["color"]),
                                        ft.Text(
                                            action.upper(),
                                            color=config["color"],
                                            size=12,
                                            weight=ft.FontWeight.BOLD,
                                        ),
                                    ],
                                    spacing=4,
                                ),
                                padding=ft.padding.symmetric(horizontal=8, vertical=4),
                                border_radius=theme.radius.SM,
                                bgcolor=ft.Colors.with_opacity(0.1, config["color"]),
                            ),
                            ft.Text(
                                f"{start_time} - {end_time}",
                                color=theme.text_secondary,
                                size=12,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    ft.Divider(height=1, color=theme.border),
                    ft.Row(
                        controls=[
                            ft.Column(
                                controls=[
                                    ft.Text(
                                        "ORIGINAL",
                                        size=10,
                                        color=theme.text_secondary,
                                        weight=ft.FontWeight.BOLD,
                                    ),
                                    ft.Text(original_text, color=theme.text_secondary, size=14),
                                ],
                                expand=True,
                            ),
                            ft.Icon("arrow_forward", color=theme.border, size=20),
                            ft.Column(
                                controls=[
                                    ft.Text(
                                        "PROCESSED",
                                        size=10,
                                        color=config["color"],
                                        weight=ft.FontWeight.BOLD,
                                    ),
                                    ft.Text(
                                        processed_text,
                                        color=theme.text_primary,
                                        size=14,
                                        weight=ft.FontWeight.W_500,
                                    ),
                                ],
                                expand=True,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    ft.Container(
                        content=ft.Text(reason, size=12, color=theme.text_secondary, italic=True),
                        padding=ft.padding.only(top=8),
                    )
                    if reason
                    else ft.Container(),
                ],
                spacing=12,
            ),
            padding=20,
            border_radius=theme.radius.LG,
            bgcolor=theme.surface,
            border=ft.border.all(1, theme.border),
        )


class LogEntry(ft.Row):
    def __init__(
        self,
        theme: Theme,
        timestamp: str,
        level: str,
        message: str,
    ):
        level_config = {
            "INFO": {"icon": "✅", "color": theme.success},
            "WARNING": {"icon": "⚠️", "color": theme.warning},
            "ERROR": {"icon": "❌", "color": theme.error},
            "PROCESSING": {"icon": "🔄", "color": theme.primary},
        }

        config = level_config.get(level.upper(), {"icon": "🔹", "color": theme.text_secondary})

        super().__init__(
            controls=[
                ft.Text(timestamp, color=theme.text_secondary, size=12, font_family="monospace"),
                ft.Text(config["icon"], size=14),
                ft.Text(
                    message,
                    color=config["color"] if level.upper() == "ERROR" else theme.text_primary,
                    size=13,
                    selectable=True,
                    expand=True,
                ),
            ],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )


class StatsCard(GlassCard):
    def __init__(
        self,
        theme: Theme,
        icon: str,
        label: str,
        value: str,
        trend: str | None = None,
        trend_up: bool = True,
    ):
        trend_color = theme.success if trend_up else theme.error
        trend_icon = "trending_up" if trend_up else "trending_down"

        content = ft.Row(
            controls=[
                ft.Container(
                    content=ft.Icon(icon, color=theme.primary, size=24),
                    padding=10,
                    bgcolor=ft.Colors.with_opacity(0.1, theme.primary),
                    border_radius=theme.radius.MD,
                ),
                ft.Column(
                    controls=[
                        ft.Text(label, color=theme.text_secondary, size=12),
                        ft.Text(
                            value, color=theme.text_primary, size=24, weight=ft.FontWeight.BOLD
                        ),
                    ],
                    spacing=2,
                    expand=True,
                ),
            ],
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        if trend:
            content.controls.append(
                ft.Container(
                    content=ft.Row(
                        controls=[
                            ft.Icon(trend_icon, size=14, color=trend_color),
                            ft.Text(trend, color=trend_color, size=12, weight=ft.FontWeight.BOLD),
                        ],
                        spacing=2,
                    ),
                    padding=ft.padding.symmetric(horizontal=8, vertical=4),
                    border_radius=theme.radius.SM,
                    bgcolor=ft.Colors.with_opacity(0.1, trend_color),
                )
            )

        super().__init__(theme=theme, content=content, padding=16)
