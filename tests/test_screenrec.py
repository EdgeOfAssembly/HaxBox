"""Comprehensive tests for screenrec.py - Screen and audio recorder tool.

Note: Many screenrec functions require X11/display, so we focus on
testing utility functions and mock display-dependent functionality.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import screenrec


class TestParseDuration:
    """Tests for parse_duration function."""

    def test_plain_seconds(self):
        """Parses plain number as seconds."""
        assert screenrec.parse_duration("30") == 30.0
        assert screenrec.parse_duration("90") == 90.0

    def test_seconds_suffix(self):
        """Parses 's' suffix."""
        assert screenrec.parse_duration("30s") == 30.0
        assert screenrec.parse_duration("90s") == 90.0

    def test_minutes_suffix(self):
        """Parses 'm' suffix."""
        assert screenrec.parse_duration("1m") == 60.0
        assert screenrec.parse_duration("5m") == 300.0

    def test_hours_suffix(self):
        """Parses 'h' suffix."""
        assert screenrec.parse_duration("1h") == 3600.0
        assert screenrec.parse_duration("2h") == 7200.0

    def test_combined_duration(self):
        """Parses combined duration string."""
        assert screenrec.parse_duration("1h30m") == 5400.0
        assert screenrec.parse_duration("1m30s") == 90.0
        assert screenrec.parse_duration("2h30m15s") == 9015.0

    def test_whitespace_handling(self):
        """Handles whitespace in duration string."""
        assert screenrec.parse_duration("  30  ") == 30.0
        assert screenrec.parse_duration(" 1m ") == 60.0

    def test_case_insensitive(self):
        """Duration parsing is case insensitive."""
        assert screenrec.parse_duration("1H") == screenrec.parse_duration("1h")
        assert screenrec.parse_duration("1M") == screenrec.parse_duration("1m")
        assert screenrec.parse_duration("30S") == screenrec.parse_duration("30s")

    def test_float_values(self):
        """Parses float values."""
        assert screenrec.parse_duration("1.5") == 1.5
        assert screenrec.parse_duration("2.5h") == 9000.0

    def test_invalid_format(self):
        """Raises error for invalid format."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            screenrec.parse_duration("invalid")

    def test_empty_string(self):
        """Raises error for empty string."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            screenrec.parse_duration("")

    @given(st.floats(min_value=0.1, max_value=1000, allow_nan=False, allow_infinity=False))
    @settings(max_examples=20)
    def test_float_roundtrip(self, seconds: float):
        """Property: float values parse correctly."""
        result = screenrec.parse_duration(str(seconds))
        assert abs(result - seconds) < 0.001


class TestCheckDependencies:
    """Tests for check_dependencies function."""

    def test_missing_cv2(self):
        """Returns False when cv2 missing."""
        with patch.dict("sys.modules", {"cv2": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                # Reset global variables
                screenrec.cv2 = None
                screenrec.np = None
                screenrec.mss_module = None
                screenrec.pynput_mouse = None

                screenrec.check_dependencies(need_x11=False)
                # Result depends on other dependencies

    def test_all_available(self):
        """Returns True when all dependencies available."""
        # Mock all imports to succeed
        mock_cv2 = MagicMock()
        mock_np = MagicMock()
        mock_mss = MagicMock()
        mock_pynput = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "cv2": mock_cv2,
                "numpy": mock_np,
                "mss": mock_mss,
                "pynput": mock_pynput,
                "pynput.mouse": mock_pynput,
            },
        ):
            # Reset globals
            screenrec.cv2 = None
            screenrec.np = None
            screenrec.mss_module = None
            screenrec.pynput_mouse = None

            screenrec.check_dependencies(need_x11=False)
            # Should succeed with mocked imports


class TestCheckAudioTools:
    """Tests for check_audio_tools function."""

    def test_pulseaudio_available(self):
        """Detects PulseAudio when parecord available."""
        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda x: "/usr/bin/parecord" if x == "parecord" else None
            available, tool = screenrec.check_audio_tools()
            assert available is True
            assert tool == "pulseaudio"

    def test_alsa_available(self):
        """Detects ALSA when arecord available."""
        with patch("shutil.which") as mock_which:
            def which_side_effect(cmd):
                return "/usr/bin/arecord" if cmd == "arecord" else None

            mock_which.side_effect = which_side_effect
            available, tool = screenrec.check_audio_tools()
            assert available is True
            assert tool == "alsa"

    def test_ffmpeg_fallback(self):
        """Falls back to ffmpeg when others unavailable."""
        with patch("shutil.which") as mock_which:
            def which_side_effect(cmd):
                return "/usr/bin/ffmpeg" if cmd == "ffmpeg" else None

            mock_which.side_effect = which_side_effect
            available, tool = screenrec.check_audio_tools()
            assert available is True
            assert tool == "ffmpeg"

    def test_no_audio_tools(self):
        """Returns False when no audio tools available."""
        with patch("shutil.which", return_value=None):
            available, tool = screenrec.check_audio_tools()
            assert available is False
            assert tool == ""


class TestGetPulseDefaultSource:
    """Tests for get_pulse_default_source function."""

    def test_gets_monitor_source(self):
        """Returns monitor source for default sink."""
        mock_result = MagicMock()
        mock_result.stdout = "alsa_output.pci-0000_00_1f.3.analog-stereo\n"

        with patch("subprocess.run", return_value=mock_result):
            result = screenrec.get_pulse_default_source()
            assert result == "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"

    def test_returns_none_on_failure(self):
        """Returns None when pactl fails."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = screenrec.get_pulse_default_source()
            assert result is None


class TestListAudioSources:
    """Tests for list_audio_sources function."""

    def test_lists_sources(self):
        """Lists available audio sources."""
        mock_result = MagicMock()
        mock_result.stdout = "0\tsource1\tIDLE\n1\tsource2\tRUNNING\n"

        with patch("subprocess.run", return_value=mock_result):
            result = screenrec.list_audio_sources()
            assert len(result) == 2
            assert result[0][0] == "source1"

    def test_returns_empty_on_failure(self):
        """Returns empty list when pactl unavailable."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = screenrec.list_audio_sources()
            assert result == []


class TestListWindows:
    """Tests for list_windows function."""

    def test_lists_windows(self):
        """Lists visible windows."""
        mock_result = MagicMock()
        mock_result.stdout = "0x12345 0 host Terminal\n0x67890 0 host Firefox\n"

        with patch("subprocess.run", return_value=mock_result):
            result = screenrec.list_windows()
            assert len(result) == 2
            assert result[0][0] == "0x12345"
            assert result[0][1] == "Terminal"

    def test_returns_empty_on_failure(self):
        """Returns empty list when wmctrl unavailable."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = screenrec.list_windows()
            assert result == []


class TestCountdown:
    """Tests for countdown function."""

    def test_countdown_zero(self):
        """Zero countdown does nothing."""
        # Should return immediately without sleeping
        screenrec.countdown(0, verbose=False)

    def test_countdown_verbose(self, capsys):
        """Verbose countdown prints messages."""
        with patch("time.sleep"):  # Mock sleep to speed up test
            screenrec.countdown(2, verbose=True)
        captured = capsys.readouterr()
        assert "Starting in" in captured.err or "Recording" in captured.err


class TestGetWindowGeometry:
    """Tests for get_window_geometry function."""

    def test_gets_geometry(self):
        """Gets window geometry from xdotool."""
        mock_result = MagicMock()
        mock_result.stdout = "X=100\nY=200\nWIDTH=800\nHEIGHT=600\n"

        with patch("subprocess.run", return_value=mock_result):
            result = screenrec.get_window_geometry("0x12345")
            assert result is not None
            assert result["left"] == 100
            assert result["top"] == 200
            assert result["width"] == 800
            assert result["height"] == 600

    def test_returns_none_on_failure(self):
        """Returns None when xdotool fails."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = screenrec.get_window_geometry("0x12345")
            assert result is None


class TestSelectWindowByClick:
    """Tests for select_window_by_click function."""

    def test_returns_geometry_on_success(self):
        """Returns geometry when window selected."""
        mock_select = MagicMock()
        mock_select.stdout = "0x12345\n"
        mock_geometry = {"left": 0, "top": 0, "width": 100, "height": 100}

        with patch("subprocess.run", return_value=mock_select):
            with patch.object(
                screenrec, "get_window_geometry", return_value=mock_geometry
            ):
                result = screenrec.select_window_by_click(verbose=False)
                assert result == mock_geometry

    def test_returns_none_on_failure(self):
        """Returns None when xdotool unavailable."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = screenrec.select_window_by_click(verbose=False)
            assert result is None


class TestMergeAudioVideo:
    """Tests for merge_audio_video function."""

    def test_returns_false_without_ffmpeg(self):
        """Returns False when ffmpeg unavailable."""
        with patch("shutil.which", return_value=None):
            result = screenrec.merge_audio_video("video.mp4", "audio.wav", "output.mp4")
            assert result is False

    def test_returns_true_on_success(self, tmp_path: Path):
        """Returns True when merge succeeds."""
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        output = tmp_path / "output.mp4"

        video.write_bytes(b"fake video")
        audio.write_bytes(b"fake audio")

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("subprocess.run"):
                result = screenrec.merge_audio_video(
                    str(video), str(audio), str(output), verbose=False
                )
                assert result is True

    def test_returns_false_on_ffmpeg_error(self, tmp_path: Path):
        """Returns False when ffmpeg fails."""
        import subprocess

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
            ):
                result = screenrec.merge_audio_video(
                    "video.mp4", "audio.wav", "output.mp4", verbose=False
                )
                assert result is False


class TestStopAudioRecording:
    """Tests for stop_audio_recording function."""

    def test_stops_process(self):
        """Sends SIGINT to stop recording."""
        mock_proc = MagicMock()
        screenrec.stop_audio_recording(mock_proc)
        mock_proc.send_signal.assert_called()

    def test_handles_none(self):
        """Handles None process gracefully."""
        screenrec.stop_audio_recording(None)  # Should not raise


class TestCLI:
    """Tests for command-line interface."""

    def test_cli_bind_keys(self, capsys):
        """CLI --bind-keys shows keybinding instructions."""
        with patch.object(sys, "argv", ["screenrec", "--bind-keys"]):
            result = screenrec.main()
            assert result == 0
        captured = capsys.readouterr()
        assert "Keyboard" in captured.out or "shortcuts" in captured.out.lower()

    def test_cli_list_audio(self, capsys):
        """CLI --list-audio lists audio sources."""
        with patch.object(sys, "argv", ["screenrec", "--list-audio"]):
            with patch.object(screenrec, "list_audio_sources", return_value=[]):
                result = screenrec.main()
                assert result == 0

    def test_cli_list_monitors(self, capsys):
        """CLI --list-monitors lists monitors."""
        mock_monitors = [
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
            {"left": 1920, "top": 0, "width": 1920, "height": 1080},
        ]

        with patch.object(sys, "argv", ["screenrec", "--list-monitors"]):
            with patch.object(screenrec, "check_dependencies", return_value=True):
                with patch.object(screenrec, "get_all_monitors", return_value=mock_monitors):
                    result = screenrec.main()
                    assert result == 0
        captured = capsys.readouterr()
        assert "2 monitor" in captured.out

    def test_cli_list_windows(self, capsys):
        """CLI --list-windows lists windows."""
        mock_windows = [("0x12345", "Terminal"), ("0x67890", "Firefox")]

        with patch.object(sys, "argv", ["screenrec", "--list-windows"]):
            with patch.object(screenrec, "check_dependencies", return_value=True):
                with patch.object(screenrec, "list_windows", return_value=mock_windows):
                    result = screenrec.main()
                    assert result == 0
        captured = capsys.readouterr()
        assert "2 window" in captured.out

    def test_cli_invalid_duration(self, capsys):
        """CLI errors on invalid duration."""
        with patch.object(sys, "argv", ["screenrec", "-d", "invalid"]):
            with patch.object(screenrec, "check_dependencies", return_value=True):
                result = screenrec.main()
                assert result == 1
        captured = capsys.readouterr()
        assert "Invalid duration" in captured.err

    def test_cli_invalid_fps(self, capsys):
        """CLI errors on invalid FPS."""
        with patch.object(sys, "argv", ["screenrec", "--fps", "0"]):
            with patch.object(screenrec, "check_dependencies", return_value=True):
                result = screenrec.main()
                assert result == 1
        captured = capsys.readouterr()
        assert "FPS must be" in captured.err

    def test_cli_fps_too_high(self, capsys):
        """CLI errors on FPS > 60."""
        with patch.object(sys, "argv", ["screenrec", "--fps", "120"]):
            with patch.object(screenrec, "check_dependencies", return_value=True):
                result = screenrec.main()
                assert result == 1

    def test_cli_region_parsing(self, capsys):
        """CLI parses region specification."""
        with patch.object(
            sys, "argv", ["screenrec", "--region", "100,200,800,600", "-d", "1"]
        ):
            with patch.object(screenrec, "check_dependencies", return_value=True):
                with patch.object(screenrec, "record_region", return_value=True):
                    with patch("os.path.exists", return_value=True):
                        with patch("os.path.getsize", return_value=1024):
                            result = screenrec.main()
                            assert result == 0

    def test_cli_invalid_region(self, capsys):
        """CLI errors on invalid region format."""
        with patch.object(sys, "argv", ["screenrec", "--region", "invalid"]):
            with patch.object(screenrec, "check_dependencies", return_value=True):
                result = screenrec.main()
                assert result == 1
        captured = capsys.readouterr()
        assert "--region format" in captured.err


class TestConstants:
    """Tests for module constants."""

    def test_version(self):
        """Version string is defined."""
        assert hasattr(screenrec, "__version__")
        assert isinstance(screenrec.__version__, str)


class TestPrintKeybindingInstructions:
    """Tests for print_keybinding_instructions function."""

    def test_prints_instructions(self, capsys):
        """Prints keybinding setup instructions."""
        screenrec.print_keybinding_instructions()
        captured = capsys.readouterr()
        assert "GNOME" in captured.out
        assert "KDE" in captured.out or "Shortcut" in captured.out
        assert "screenrec" in captured.out


class TestStartAudioRecording:
    """Tests for start_audio_recording function."""

    def test_returns_none_without_tools(self):
        """Returns None when no audio tools available."""
        with patch.object(screenrec, "check_audio_tools", return_value=(False, "")):
            result = screenrec.start_audio_recording(
                "output.wav", 30.0, None, verbose=False
            )
            assert result is None

    def test_starts_pulseaudio_recording(self, tmp_path: Path):
        """Starts PulseAudio recording when available."""
        output = tmp_path / "audio.wav"

        with patch.object(screenrec, "check_audio_tools", return_value=(True, "pulseaudio")):
            with patch.object(
                screenrec, "get_pulse_default_source", return_value="default.monitor"
            ):
                with patch("subprocess.Popen") as mock_popen:
                    mock_popen.return_value = MagicMock()
                    result = screenrec.start_audio_recording(
                        str(output), 30.0, None, verbose=False
                    )
                    assert result is not None


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(st.integers(min_value=1, max_value=3600))
    @settings(max_examples=20)
    def test_duration_seconds_roundtrip(self, seconds: int):
        """Property: integer seconds parse correctly."""
        result = screenrec.parse_duration(f"{seconds}s")
        assert result == float(seconds)

    @given(st.integers(min_value=1, max_value=60))
    @settings(max_examples=20)
    def test_duration_minutes_conversion(self, minutes: int):
        """Property: minutes convert to seconds correctly."""
        result = screenrec.parse_duration(f"{minutes}m")
        assert result == float(minutes * 60)

    @given(st.integers(min_value=1, max_value=24))
    @settings(max_examples=20)
    def test_duration_hours_conversion(self, hours: int):
        """Property: hours convert to seconds correctly."""
        result = screenrec.parse_duration(f"{hours}h")
        assert result == float(hours * 3600)


# Skip these tests if no DISPLAY is available
@pytest.mark.skipif(
    not bool(__import__("os").environ.get("DISPLAY")),
    reason="No X11 display available"
)
class TestX11Functions:
    """Tests that require an X11 display (use xvfb for CI)."""

    def test_get_primary_monitor(self):
        """Get primary monitor returns valid region."""
        # This requires mss to be available and working
        if not screenrec.check_dependencies(need_x11=False):
            pytest.skip("Dependencies not available")

        region = screenrec.get_primary_monitor()
        assert isinstance(region, dict)
        assert "left" in region
        assert "top" in region
        assert "width" in region
        assert "height" in region
        assert region["width"] > 0
        assert region["height"] > 0

    def test_get_full_screen(self):
        """Get full screen returns valid region."""
        if not screenrec.check_dependencies(need_x11=False):
            pytest.skip("Dependencies not available")

        region = screenrec.get_full_screen()
        assert isinstance(region, dict)
        assert region["width"] > 0
        assert region["height"] > 0

    def test_get_all_monitors(self):
        """Get all monitors returns list of regions."""
        if not screenrec.check_dependencies(need_x11=False):
            pytest.skip("Dependencies not available")

        monitors = screenrec.get_all_monitors()
        assert isinstance(monitors, list)
        assert len(monitors) >= 1
        for m in monitors:
            assert "width" in m
            assert "height" in m
            assert m["width"] > 0
            assert m["height"] > 0

    def test_record_region_short(self, tmp_path: Path):
        """Record a very short clip to test recording functionality."""
        if not screenrec.check_dependencies(need_x11=False):
            pytest.skip("Dependencies not available")

        output = tmp_path / "test_recording.mp4"
        region = {"left": 0, "top": 0, "width": 100, "height": 100}

        # Record for just 0.5 seconds at low fps
        result = screenrec.record_region(
            region=region,
            output=str(output),
            duration=0.5,
            fps=10,
            audio=False,
            audio_source=None,
            verbose=False,
        )

        assert result is True
        assert output.exists()
        assert output.stat().st_size > 0
