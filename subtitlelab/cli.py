# -*- coding: utf-8 -*-
"""
SubtitleLab CLI
Interact with SubtitleLab via command line.
"""

import sys
import os
import argparse
import asyncio
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.logging import RichHandler
import logging

from .core.config import AppConfig, CONFIG_FILE
from .core.processor import SubtitleProcessor, ProcessorCallbacks
from .core.models import BatchResult, ProcessingAction

# Initialize Rich Console
console = Console()

# Configure logging
logging.basicConfig(
    level="ERROR",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_path=False)],
)


def run_init_wizard(config: AppConfig) -> None:
    """Interactive configuration wizard."""
    console.print(Panel.fit("SubtitleLab Configuration Wizard", style="bold blue"))

    console.print("\n[bold]LLM Settings[/bold]")

    provider = Prompt.ask(
        "Select Provider", choices=["openai", "deepseek", "ollama", "custom"], default="openai"
    )

    if provider == "openai":
        base_url = Prompt.ask("API Base URL", default="https://api.openai.com/v1")
        model = Prompt.ask("Model", default="gpt-4o")
    elif provider == "deepseek":
        base_url = Prompt.ask("API Base URL", default="https://api.deepseek.com")
        model = Prompt.ask("Model", default="deepseek-chat")
    elif provider == "ollama":
        base_url = Prompt.ask("API Base URL", default="http://localhost:11434/v1")
        model = Prompt.ask("Model", default="llama3")
    else:
        base_url = Prompt.ask("API Base URL")
        model = Prompt.ask("Model")

    api_key = Prompt.ask("API Key", password=True)

    config.llm.api_base = base_url
    config.llm.api_key = api_key
    config.llm.model = model

    console.print("\n[bold]Processing Settings[/bold]")
    config.processing.concurrency = IntPrompt.ask("Concurrency (simultaneous batches)", default=3)
    config.processing.window_size = IntPrompt.ask("Window Size (lines per batch)", default=20)
    config.processing.allow_dynamic_window = Confirm.ask(
        "Enable Auto-Chunking (Recommended)", default=True
    )

    config.save()
    console.print(f"\n[green]Configuration saved to {CONFIG_FILE}[/green]")


async def process_file(file_path: Path, config: AppConfig, output_path: Optional[Path] = None):
    """Process a subtitle file."""
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        return

    processor = SubtitleProcessor(config)

    # Progress Bar Setup
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    task_id = progress.add_task("Processing...", total=100)

    def on_progress(p: float, msg: str):
        progress.update(task_id, completed=p * 100, description=msg)

    def on_batch_complete(result: BatchResult):
        # We can print detailed logs for each batch if verbose mode is on
        # For now, just let the progress bar handle it
        pass

    def on_error(error: Exception, context: str):
        console.print(f"[red]Error in {context}: {error}[/red]")

    processor.callbacks = ProcessorCallbacks(
        on_progress=on_progress, on_batch_complete=on_batch_complete, on_error=on_error
    )

    console.print(f"[bold cyan]Starting processing for: {file_path.name}[/bold cyan]")

    try:
        processor.load_subtitles(file_path)
    except Exception as e:
        console.print(f"[bold red]Failed to load file:[/bold red] {e}")
        return

    with progress:
        try:
            stats = await processor.process()
        except Exception as e:
            console.print(f"[bold red]Fatal Error:[/bold red] {e}")
            return

    # Show Summary Table
    table = Table(title="Processing Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Entries", str(stats.total_entries))
    table.add_row("Processed Entries", str(stats.processed_entries))
    table.add_row("Merges", str(stats.merged_count))
    table.add_row("Deletions", str(stats.deleted_count))
    table.add_row("Corrections", str(stats.corrected_count))
    table.add_row("Time Elapsed", f"{stats.elapsed_time:.2f}s")

    console.print(table)

    # Save output
    if not output_path:
        output_path = file_path.with_name(f"{file_path.stem}_processed.srt")

    console.print(f"Saving to: [bold]{output_path}[/bold]")
    processor.save_results(output_path)
    console.print("[bold green]Done![/bold green]")


def show_config(config: AppConfig):
    """Display current configuration."""
    table = Table(title="Current Configuration")
    table.add_column("Section", style="cyan")
    table.add_column("Key", style="yellow")
    table.add_column("Value", style="white")

    table.add_row("LLM", "API Base", config.llm.api_base)
    table.add_row("LLM", "Model", config.llm.model)
    # Mask API Key
    key = config.llm.api_key
    masked_key = f"{key[:3]}...{key[-4:]}" if len(key) > 8 else "***"
    table.add_row("LLM", "API Key", masked_key)

    table.add_row("Processing", "Concurrency", str(config.processing.concurrency))
    table.add_row("Processing", "Auto Chunking", str(config.processing.allow_dynamic_window))

    console.print(table)
    console.print(f"Config file: {CONFIG_FILE}")


def main():
    parser = argparse.ArgumentParser(description="SubtitleLab CLI - Intelligent Subtitle Processor")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Init Command
    subparsers.add_parser("init", help="Initialize configuration wizard")

    # Config Command
    subparsers.add_parser("config", help="Show current configuration")

    # Process Command
    proc_parser = subparsers.add_parser("process", help="Process a subtitle file")
    proc_parser.add_argument("file", help="Path to subtitle file (.srt, .ass)")
    proc_parser.add_argument("-o", "--output", help="Output path (optional)")

    args = parser.parse_args()

    # Load config
    config = AppConfig.load()

    if args.command == "init":
        run_init_wizard(config)
    elif args.command == "config":
        show_config(config)
    elif args.command == "process":
        if not config.llm.api_key:
            console.print(
                "[yellow]Warning: API Key not found. Running init wizard first...[/yellow]"
            )
            run_init_wizard(config)

        file_path = Path(args.file)
        output_path = Path(args.output) if args.output else None

        asyncio.run(process_file(file_path, config, output_path))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
