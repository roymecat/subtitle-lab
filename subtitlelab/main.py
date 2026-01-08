"""SubtitleLab - Intelligent Subtitle Post-Processing Tool"""

import asyncio
import sys


def main():
    try:
        import flet as ft
        from subtitlelab.gui.app import SubtitleLabApp
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Please install dependencies: pip install flet openai pysubs2 tenacity")
        sys.exit(1)

    app = SubtitleLabApp()
    ft.app(target=app.main)


if __name__ == "__main__":
    main()
