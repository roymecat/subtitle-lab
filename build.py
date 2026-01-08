#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SubtitleLab Build Script (CLI Edition)
Creates Windows executable using PyInstaller.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def clean_build():
    """Remove previous build artifacts."""
    dirs_to_remove = ["build", "dist", "__pycache__"]
    files_to_remove = ["*.spec"]

    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            print(f"Removing {dir_name}/...")
            shutil.rmtree(dir_name)

    for pattern in files_to_remove:
        for f in Path(".").glob(pattern):
            print(f"Removing {f}...")
            f.unlink()


def build_executable():
    """Build the Windows executable using PyInstaller."""

    app_name = "SubtitleLab"
    main_script = "subtitlelab/main.py"  # Use main.py to preserve package structure

    if not os.path.exists(main_script):
        print(f"Error: {main_script} not found!")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        app_name,
        "--onefile",  # Single executable
        "--noconfirm",  # Overwrite without asking
        "--console",  # CLI application
        # Hidden imports
        "--hidden-import",
        "rich",
        "--hidden-import",
        "httpx",
        "--hidden-import",
        "pysubs2",
        "--hidden-import",
        "tiktoken",
        "--hidden-import",
        "openai",
        "--hidden-import",
        "pydantic",
        "--hidden-import",
        "tenacity",
        "--hidden-import",
        "aiofiles",
        # Collect metadata
        "--collect-all",
        "rich",
        "--collect-all",
        "openai",
        "--collect-all",
        "httpx",
        "--collect-all",
        "pysubs2",
        # Exclude unnecessary modules
        "--exclude-module",
        "matplotlib",
        "--exclude-module",
        "numpy",
        "--exclude-module",
        "pandas",
        "--exclude-module",
        "PIL",
        "--exclude-module",
        "tkinter",
        "--exclude-module",
        "PyQt6",
        "--exclude-module",
        "PySide6",
        "--exclude-module",
        "textual",
        # Main script
        main_script,
    ]

    print("Building SubtitleLab CLI executable...")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("=" * 50)
        print("Build completed successfully!")
        print(f"Executable: dist/{app_name}.exe")
        print("=" * 50)
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error code {e.returncode}")
        sys.exit(1)


def main():
    os.chdir(Path(__file__).parent)

    if "--clean" in sys.argv:
        clean_build()

    if "--clean-only" in sys.argv:
        clean_build()
        return

    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

    build_executable()


if __name__ == "__main__":
    main()
