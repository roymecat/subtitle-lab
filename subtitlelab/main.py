"""SubtitleLab - Intelligent Subtitle Post-Processing Tool"""

import sys
import os
import traceback
from pathlib import Path


def setup_error_logging():
    """Setup error logging to file for debugging packaged app."""
    if getattr(sys, "frozen", False):
        log_dir = Path(os.path.expanduser("~")) / ".subtitlelab"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "error.log"
        return log_file
    return None


def main():
    log_file = setup_error_logging()

    try:
        import flet as ft
        from subtitlelab.gui.app import SubtitleLabApp

        app = SubtitleLabApp()
        ft.app(target=app.main)

    except Exception as e:
        error_msg = f"SubtitleLab Error:\n{traceback.format_exc()}"

        if log_file:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(error_msg)

        if sys.platform == "win32":
            try:
                import ctypes

                ctypes.windll.user32.MessageBoxW(
                    0,
                    f"启动失败，错误日志已保存到:\n{log_file}\n\n{str(e)}",
                    "SubtitleLab Error",
                    0x10,
                )
            except:
                pass

        print(error_msg, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
