#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SubtitleLab Build Script
Creates Windows executable using PyInstaller.

Usage:
    python build.py          # Build executable
    python build.py --clean  # Clean build artifacts first
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

    # PyInstaller options
    app_name = "SubtitleLab"
    main_script = "subtitlelab/main.py"

    # Check if main script exists
    if not os.path.exists(main_script):
        print(f"Error: {main_script} not found!")
        sys.exit(1)

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        app_name,
        "--windowed",  # No console window
        "--onefile",  # Single executable
        "--noconfirm",  # Overwrite without asking
        # Add data files
        "--add-data",
        "subtitlelab/gui/translations:subtitlelab/gui/translations",
        # Hidden imports for PyQt6
        "--hidden-import",
        "PyQt6.QtCore",
        "--hidden-import",
        "PyQt6.QtGui",
        "--hidden-import",
        "PyQt6.QtWidgets",
        "--hidden-import",
        "PyQt6.sip",
        # Hidden imports for other dependencies
        "--hidden-import",
        "httpx",
        "--hidden-import",
        "pysubs2",
        "--hidden-import",
        "tiktoken",
        "--hidden-import",
        "tiktoken_ext",
        "--hidden-import",
        "tiktoken_ext.openai_public",
        # Exclude unnecessary modules to reduce size
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
        # Main script
        main_script,
    ]

    print("Building SubtitleLab executable...")
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
    except FileNotFoundError:
        print("Error: PyInstaller not found. Install it with:")
        print("  pip install pyinstaller")
        sys.exit(1)


def main():
    # Change to script directory
    os.chdir(Path(__file__).parent)

    # Parse arguments
    if "--clean" in sys.argv:
        clean_build()

    if "--clean-only" in sys.argv:
        clean_build()
        return

    # Check dependencies
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

    build_executable()


if __name__ == "__main__":
    main()
