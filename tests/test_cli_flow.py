import asyncio
import sys
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from subtitlelab.cli import main
from subtitlelab.core.config import AppConfig, CONFIG_FILE
from subtitlelab.core.models import ProcessingStats


def test_init_flow():
    """Test 'init' command."""
    print("\n=== Testing Init Flow ===")

    # Backup existing config if any
    if CONFIG_FILE.exists():
        shutil.copy(CONFIG_FILE, CONFIG_FILE.with_suffix(".bak"))
        CONFIG_FILE.unlink()

    try:
        # Mock user input
        with patch(
            "rich.prompt.Prompt.ask",
            side_effect=["custom", "http://test.api", "test-model", "sk-test-key"],
        ):
            with patch("rich.prompt.IntPrompt.ask", return_value=3):
                with patch("rich.prompt.Confirm.ask", return_value=True):
                    with patch("sys.argv", ["subtitlelab", "init"]):
                        main()

        # Verify config created
        assert CONFIG_FILE.exists()
        config = AppConfig.load()
        assert config.llm.api_base == "http://test.api"
        assert config.llm.api_key == "sk-test-key"
        print("✅ Init Flow Passed")

    finally:
        # Restore backup
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
        if CONFIG_FILE.with_suffix(".bak").exists():
            shutil.move(CONFIG_FILE.with_suffix(".bak"), CONFIG_FILE)


def test_process_flow():
    """Test 'process' command."""
    print("\n=== Testing Process Flow ===")

    # Create dummy subtitle file
    dummy_srt = Path("test_video.srt")
    dummy_srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello World\n", encoding="utf-8")

    try:
        # Mock Processor.process to avoid actual API calls
        mock_stats = ProcessingStats(
            total_entries=1, processed_entries=1, merged_count=0, deleted_count=0, corrected_count=0
        )

        with patch(
            "subtitlelab.core.processor.SubtitleProcessor.process", new_callable=MagicMock
        ) as mock_process:
            mock_process.return_value = mock_stats

            # Mock process to be awaitable
            async def async_mock():
                return mock_stats

            mock_process.side_effect = async_mock

            # Mock load_subtitles
            with patch("subtitlelab.core.processor.SubtitleProcessor.load_subtitles") as mock_load:
                with patch("sys.argv", ["subtitlelab", "process", str(dummy_srt)]):
                    # Ensure we have a config with key (or mock load)
                    with patch("subtitlelab.core.config.AppConfig.load") as mock_config_load:
                        cfg = AppConfig()
                        cfg.llm.api_key = "test"
                        mock_config_load.return_value = cfg

                        main()

        print("✅ Process Flow Passed")

    finally:
        if dummy_srt.exists():
            dummy_srt.unlink()


if __name__ == "__main__":
    try:
        test_init_flow()
        test_process_flow()
        print("\n🎉 All Tests Passed!")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        sys.exit(1)
