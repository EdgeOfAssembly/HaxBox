#!/usr/bin/env python3
"""screenrec - Flexible screen and audio recorder.

Record screen to video with multiple capture modes (like gnome-screenshot):
- Full screen capture (like PrtScn)
- Window selection - click on a window (like Alt+PrtScn)
- Area selection - draw a rectangle (like Shift+PrtScn)
- Optional audio capture

Author: EdgeOfAssembly
Email: haxbox2000@gmail.com
License: GPLv3 / Commercial
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Callable

__version__ = "1.0.0"

# Lazy imports for optional dependencies - typed as Any since they're optional
cv2: Optional[Any] = None
np: Optional[Any] = None
mss_module: Optional[Callable[..., Any]] = None
pynput_mouse: Optional[Any] = None


def check_dependencies(need_x11: bool = True) -> bool:
    """Check and import required dependencies.

    Args:
        need_x11: Whether X11 dependencies (pynput) are needed.

    Returns:
        True if all required dependencies are available, False otherwise.
    """
    global cv2, np, mss_module, pynput_mouse

    missing: List[str] = []

    try:
        import cv2 as _cv2
        cv2 = _cv2
    except ImportError:
        missing.append("opencv-python-headless")

    try:
        import numpy as _np
        np = _np
    except ImportError:
        missing.append("numpy")

    try:
        from mss import mss as _mss
        mss_module = _mss
    except ImportError:
        missing.append("mss")

    if need_x11:
        try:
            from pynput import mouse as _mouse
            pynput_mouse = _mouse
        except ImportError as e:
            err_str = str(e).lower()
            if "not supported" in err_str or "x connection" in err_str or "display" in err_str:
                print("Error: This tool requires an X11 display.", file=sys.stderr)
                print("Set DISPLAY environment variable or run in a desktop session.", file=sys.stderr)
                sys.exit(1)
            missing.append("pynput")

    if missing:
        print(f"Error: Missing dependencies: {', '.join(missing)}", file=sys.stderr)
        print(f"Install with: pip install {' '.join(missing)}", file=sys.stderr)
        return False

    return True


def check_audio_tools() -> Tuple[bool, str]:
    """Check for audio recording tools. Returns (available, tool_name)."""
    # Check for PulseAudio tools
    if shutil.which("parecord"):
        return True, "pulseaudio"
    # Check for ALSA tools
    if shutil.which("arecord"):
        return True, "alsa"
    # Check for ffmpeg with audio support
    if shutil.which("ffmpeg"):
        return True, "ffmpeg"
    return False, ""


def get_pulse_default_source() -> Optional[str]:
    """Get default PulseAudio source (for desktop audio capture)."""
    try:
        # Try to get the monitor of the default sink (captures desktop audio)
        result = subprocess.run(
            ["pactl", "get-default-sink"],
            capture_output=True, text=True, check=True
        )
        default_sink = result.stdout.strip()
        return f"{default_sink}.monitor"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def list_audio_sources() -> List[Tuple[str, str]]:
    """List available audio sources."""
    sources = []
    try:
        result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True, text=True, check=True
        )
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split("\t")
                if len(parts) >= 2:
                    sources.append((parts[1], parts[1]))
    except (subprocess.CalledProcessError, FileNotFoundError):
        # PulseAudio tools unavailable or failed, return empty list
        pass
    return sources


def parse_duration(s: str) -> float:
    """Parse duration string (e.g., '30', '1m30s', '90s', '1h')."""
    s = s.strip().lower()
    
    # Try plain number (seconds)
    try:
        return float(s)
    except ValueError:
        pass
    
    total = 0.0
    
    # Match hours
    match = re.search(r'(\d+(?:\.\d+)?)\s*h', s)
    if match:
        total += float(match.group(1)) * 3600
    
    # Match minutes  
    match = re.search(r'(\d+(?:\.\d+)?)\s*m(?!s)', s)
    if match:
        total += float(match.group(1)) * 60
    
    # Match seconds
    match = re.search(r'(\d+(?:\.\d+)?)\s*s', s)
    if match:
        total += float(match.group(1))
    
    if total > 0:
        return total

    raise ValueError(f"Invalid duration format: {s}")


def get_primary_monitor() -> Dict[str, int]:
    """Get primary monitor region.

    Returns:
        Dictionary with keys 'left', 'top', 'width', 'height' for the primary monitor.

    Raises:
        RuntimeError: If mss is not available.
    """
    if mss_module is None:
        raise RuntimeError("mss module not available")
    with mss_module() as sct:
        if len(sct.monitors) > 1:
            return dict(sct.monitors[1])
        return dict(sct.monitors[0])


def get_full_screen() -> Dict[str, int]:
    """Get full virtual screen (all monitors combined).

    Returns:
        Dictionary with keys 'left', 'top', 'width', 'height' for the full screen.

    Raises:
        RuntimeError: If mss is not available.
    """
    if mss_module is None:
        raise RuntimeError("mss module not available")
    with mss_module() as sct:
        return dict(sct.monitors[0])


def get_all_monitors() -> List[Dict[str, int]]:
    """Get all monitor regions.

    Returns:
        List of dictionaries with monitor geometry information.

    Raises:
        RuntimeError: If mss is not available.
    """
    if mss_module is None:
        raise RuntimeError("mss module not available")
    with mss_module() as sct:
        return [dict(m) for m in sct.monitors[1:]]


def get_window_geometry(window_id: str) -> Optional[Dict[str, int]]:
    """Get window geometry using xdotool."""
    try:
        result = subprocess.run(
            ["xdotool", "getwindowgeometry", "--shell", window_id],
            capture_output=True, text=True, check=True
        )
        
        geometry = {}
        for line in result.stdout.strip().split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                geometry[key] = int(value)
        
        return {
            "left": geometry.get("X", 0),
            "top": geometry.get("Y", 0),
            "width": geometry.get("WIDTH", 100),
            "height": geometry.get("HEIGHT", 100)
        }
    except (subprocess.CalledProcessError, FileNotFoundError, KeyError, ValueError):
        return None


def select_window_by_click(verbose: bool = False) -> Optional[Dict[str, int]]:
    """Let user click on a window to select it."""
    try:
        if verbose:
            print("Click on the window you want to record...", file=sys.stderr)
        
        result = subprocess.run(
            ["xdotool", "selectwindow"],
            capture_output=True, text=True, check=True
        )
        window_id = result.stdout.strip()
        
        if verbose:
            print(f"Selected window ID: {window_id}", file=sys.stderr)
        
        return get_window_geometry(window_id)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: xdotool required. Install: apt install xdotool", file=sys.stderr)
        return None


def list_windows() -> List[Tuple[str, str]]:
    """List all visible windows.

    Returns:
        List of tuples (window_id, window_title) for visible windows.
    """
    try:
        result = subprocess.run(
            ["wmctrl", "-l"],
            capture_output=True, text=True, check=True
        )

        windows: List[Tuple[str, str]] = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split(None, 3)
                if len(parts) >= 4:
                    windows.append((parts[0], parts[3]))
        return windows
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


# Global state for rectangle selection
_rect_selection: Dict[str, int] = {"start_x": 0, "start_y": 0, "end_x": 0, "end_y": 0}


def select_rectangle(verbose: bool = False) -> Optional[Dict[str, int]]:
    """Let user draw a rectangle to select region.

    Args:
        verbose: Whether to print progress messages.

    Returns:
        Dictionary with 'left', 'top', 'width', 'height' keys, or None if selection failed.

    Raises:
        RuntimeError: If pynput is not available.
    """
    global _rect_selection
    _rect_selection = {"start_x": 0, "start_y": 0, "end_x": 0, "end_y": 0}

    if pynput_mouse is None:
        raise RuntimeError("pynput module not available")

    if verbose:
        print("Click and drag to select recording region...", file=sys.stderr)

    def on_click(x: int, y: int, button: Any, pressed: bool) -> Optional[bool]:
        if button != pynput_mouse.Button.left:
            return None

        if pressed:
            _rect_selection["start_x"] = x
            _rect_selection["start_y"] = y
            if verbose:
                print(f"  Start: ({x}, {y})", file=sys.stderr)
        else:
            _rect_selection["end_x"] = x
            _rect_selection["end_y"] = y
            if verbose:
                print(f"  End: ({x}, {y})", file=sys.stderr)
            return False
        return None

    with pynput_mouse.Listener(on_click=on_click) as listener:
        listener.join()
    
    left = min(_rect_selection["start_x"], _rect_selection["end_x"])
    top = min(_rect_selection["start_y"], _rect_selection["end_y"])
    width = abs(_rect_selection["end_x"] - _rect_selection["start_x"])
    height = abs(_rect_selection["end_y"] - _rect_selection["start_y"])
    
    if width < 10 or height < 10:
        return None
    
    return {"left": left, "top": top, "width": width, "height": height}


def countdown(seconds: int, verbose: bool = False) -> None:
    """Show countdown/delay before recording starts."""
    if seconds <= 0:
        return
    
    for i in range(seconds, 0, -1):
        if verbose:
            print(f"\rStarting in {i}...", end="", file=sys.stderr)
        time.sleep(1)
    if verbose:
        print("\rRecording!     ", file=sys.stderr)


def print_keybinding_instructions() -> None:
    """Print instructions for setting up custom keyboard shortcuts."""
    script_path = os.path.abspath(__file__)
    
    print("""
screenrec - Custom Keyboard Shortcut Setup
===========================================

If gnome-screenshot already uses the default keys (PrtScn, Alt+PrtScn, 
Shift+PrtScn), you can bind screenrec to different keys.

GNOME (Settings > Keyboard > Shortcuts > Custom Shortcuts):
-----------------------------------------------------------
Add these custom shortcuts:

  Name: Screen Record (Full)
  Command: {script} --fullscreen --delay 3 -d 60 -v
  Shortcut: Super+Print  (or your choice)

  Name: Screen Record (Window)  
  Command: {script} --window --delay 3 -d 60 -v
  Shortcut: Super+Alt+Print  (or your choice)

  Name: Screen Record (Area)
  Command: {script} --select -d 60 -v
  Shortcut: Super+Shift+Print  (or your choice)

KDE Plasma (System Settings > Shortcuts > Custom Shortcuts):
------------------------------------------------------------
Add new Global Shortcuts with the commands above.

XFCE (Settings > Keyboard > Application Shortcuts):
---------------------------------------------------
Add the commands above with your preferred key combinations.

i3/Sway (config file):
----------------------
bindsym $mod+Print exec {script} --fullscreen --delay 3 -d 60
bindsym $mod+Shift+Print exec {script} --select -d 60
bindsym $mod+Mod1+Print exec {script} --window --delay 3 -d 60

Command-line with xdotool (generic X11):
----------------------------------------
# You can use xbindkeys or similar tools:
# ~/.xbindkeysrc:
"{script} --fullscreen --delay 3 -d 60"
  Mod4 + Print

"{script} --window --delay 3 -d 60"
  Mod4 + Mod1 + Print

"{script} --select -d 60"
  Mod4 + Shift + Print

Tips:
-----
- Use --delay to give yourself time to switch windows
- Use -v (--verbose) to see countdown and progress
- Adjust -d (--duration) for longer/shorter recordings
- Add --audio to capture desktop sound
- Output goes to /tmp/screenrec.mp4 by default (use -o to change)
""".format(script=script_path))


def start_audio_recording(
    output_file: str, duration: float,
    audio_source: Optional[str], verbose: bool
) -> Optional["subprocess.Popen[bytes]"]:
    """Start audio recording in background.

    Args:
        output_file: Path to save the audio file.
        duration: Recording duration in seconds.
        audio_source: Optional PulseAudio source name.
        verbose: Whether to print progress messages.

    Returns:
        Subprocess Popen object if successful, None otherwise.
    """
    audio_available, tool = check_audio_tools()
    if not audio_available:
        if verbose:
            print("Warning: No audio recording tools found", file=sys.stderr)
        return None

    source = audio_source or get_pulse_default_source()

    try:
        cmd: List[str]
        if tool == "pulseaudio" and source:
            cmd = [
                "parecord", "--file-format=wav",
                "-d", source,
                output_file
            ]
        elif tool == "ffmpeg":
            cmd = [
                "ffmpeg", "-y", "-f", "pulse",
                "-i", source or "default",
                "-t", str(duration),
                "-ac", "2", "-ar", "44100",
                output_file
            ]
        else:
            cmd = ["arecord", "-f", "cd", "-t", "wav", output_file]

        if verbose:
            print(f"Starting audio: {' '.join(cmd)}", file=sys.stderr)
        
        proc: "subprocess.Popen[bytes]" = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return proc
    except Exception as e:
        if verbose:
            print(f"Warning: Could not start audio recording: {e}", file=sys.stderr)
        return None


def stop_audio_recording(proc: Optional["subprocess.Popen[bytes]"]) -> None:
    """Stop audio recording process.

    Args:
        proc: Subprocess Popen object to stop.
    """
    if proc:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def merge_audio_video(video_file: str, audio_file: str, output_file: str,
                      verbose: bool = False) -> bool:
    """Merge audio and video files using ffmpeg.

    Args:
        video_file: Path to video file.
        audio_file: Path to audio file.
        output_file: Path for merged output file.
        verbose: Whether to print progress messages.

    Returns:
        True if merge succeeded, False otherwise.
    """
    if not shutil.which("ffmpeg"):
        if verbose:
            print("Warning: ffmpeg not found, cannot merge audio", file=sys.stderr)
        return False
    
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_file,
            "-i", audio_file,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_file
        ]
        
        if verbose:
            print("Merging audio/video...", file=sys.stderr)
        
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"Warning: Failed to merge audio: {e}", file=sys.stderr)
        return False


def record_region(region: Dict[str, int], output: str, duration: float,
                  fps: float, audio: bool, audio_source: Optional[str],
                  verbose: bool = False) -> bool:
    """Record screen region to video file, optionally with audio.

    Args:
        region: Dictionary with 'left', 'top', 'width', 'height' keys.
        output: Path for output video file.
        duration: Recording duration in seconds.
        fps: Frames per second.
        audio: Whether to record audio.
        audio_source: Optional PulseAudio source name.
        verbose: Whether to print progress messages.

    Returns:
        True if recording succeeded, False otherwise.
    """
    if cv2 is None or np is None or mss_module is None:
        print("Error: Required dependencies not available", file=sys.stderr)
        return False

    width = max(1, region["width"])
    height = max(1, region["height"])

    # Ensure even dimensions for video encoding
    width = width - (width % 2)
    height = height - (height % 2)
    region["width"] = width
    region["height"] = height

    if verbose:
        print(f"Recording: {width}x{height} at ({region['left']}, {region['top']})",
              file=sys.stderr)
        print(f"Duration: {duration}s, FPS: {fps}, Audio: {audio}", file=sys.stderr)

    # Setup output paths
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # If audio, use temp files for separate recording
    temp_video: Optional[str] = None
    temp_audio: Optional[str] = None
    if audio:
        temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        video_output = temp_video
    else:
        video_output = output

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"Error: Could not create video file: {video_output}", file=sys.stderr)
        return False

    # Start audio recording
    audio_proc: Optional["subprocess.Popen[bytes]"] = None
    if audio and temp_audio is not None:
        audio_proc = start_audio_recording(temp_audio, duration, audio_source, verbose)

    frame_interval = 1.0 / fps
    frames_recorded = 0
    start_time = time.time()

    try:
        with mss_module() as sct:
            while True:
                loop_start = time.time()
                elapsed = loop_start - start_time
                
                if elapsed >= duration:
                    break
                
                # Capture frame
                img = sct.grab(region)
                frame = np.array(img)
                frame_bgr = frame[:, :, :3]
                
                if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
                    frame_bgr = cv2.resize(frame_bgr, (width, height))
                
                writer.write(frame_bgr)
                frames_recorded += 1
                
                # Progress
                if verbose and frames_recorded % int(fps) == 0:
                    remaining = duration - elapsed
                    print(f"\r  {remaining:.0f}s remaining", end="", file=sys.stderr)
                
                # Maintain frame rate
                process_time = time.time() - loop_start
                sleep_time = frame_interval - process_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        if verbose:
            print(file=sys.stderr)
        
    except KeyboardInterrupt:
        if verbose:
            print("\nRecording stopped", file=sys.stderr)
    finally:
        writer.release()
        if audio_proc:
            stop_audio_recording(audio_proc)
        
        # Cleanup temp files on error - moved inside finally block
        # Merge audio and video if needed
        if audio and temp_video and temp_audio:
            try:
                if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 0:
                    if merge_audio_video(temp_video, temp_audio, output, verbose):
                        os.unlink(temp_video)
                        os.unlink(temp_audio)
                    else:
                        # Fallback: use video only
                        shutil.move(temp_video, output)
                        if os.path.exists(temp_audio):
                            os.unlink(temp_audio)
                else:
                    shutil.move(temp_video, output)
                    if os.path.exists(temp_audio):
                        os.unlink(temp_audio)
            except Exception:
                # Ensure temp files are cleaned up even if merge fails
                if temp_video and os.path.exists(temp_video):
                    os.unlink(temp_video)
                if temp_audio and os.path.exists(temp_audio):
                    os.unlink(temp_audio)
                raise
    
    if verbose:
        actual_duration = time.time() - start_time
        actual_fps = frames_recorded / actual_duration if actual_duration > 0 else 0
        print(f"Recorded {frames_recorded} frames ({actual_fps:.1f} fps)", file=sys.stderr)
    
    return frames_recorded > 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Flexible screen and audio recorder (gnome-screenshot style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Capture Modes (like gnome-screenshot):
  --fullscreen       Record entire screen (like PrtScn)
  --window           Click on a window to record (like Alt+PrtScn)
  --select           Draw rectangle to select area (like Shift+PrtScn)
  --monitor N        Record specific monitor N (1=primary)
  --window-id ID     Record window with specific X11 ID
  --region X,Y,W,H   Record specific coordinates

Timing:
  --delay SEC        Wait SEC seconds before starting capture
  --duration TIME    Record for TIME (e.g., 30, 30s, 1m, 1h)

Audio:
  --audio            Enable audio recording (desktop audio)
  --audio-source SRC Use specific PulseAudio source

Keyboard Shortcuts:
  --bind-keys        Show commands to bind custom keyboard shortcuts
                     (useful if gnome-screenshot uses default keys)

Examples:
  # Draw area to record (default), 30 seconds
  screenrec
  
  # Record full screen with 3 second delay
  screenrec --fullscreen --delay 3 -d 60
  
  # Click on a window, wait 5 seconds, record for 2 minutes
  screenrec --window --delay 5 -d 2m
  
  # Record selected area with audio
  screenrec --select --audio -d 30
  
  # Record primary monitor for 2 minutes
  screenrec --monitor 1 -d 2m
  
  # Show how to set up keyboard shortcuts
  screenrec --bind-keys
"""
    )
    
    # Output options
    parser.add_argument("-o", "--output", default="/tmp/screenrec.mp4",
                        help="Output file (default: /tmp/screenrec.mp4)")
    parser.add_argument("-d", "--duration", default="30",
                        help="Recording duration: 30, 30s, 1m, 1h (default: 30)")
    parser.add_argument("--delay", type=int, default=0, metavar="SEC",
                        help="Delay in seconds before recording starts")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Frames per second (default: 30, max: 60)")
    
    # Capture mode (mutually exclusive)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fullscreen", action="store_true",
                      help="Record entire screen (like PrtScn)")
    mode.add_argument("--monitor", type=int, metavar="N",
                      help="Record monitor N (1=primary)")
    mode.add_argument("--window", action="store_true",
                      help="Click to select window (like Alt+PrtScn)")
    mode.add_argument("--window-id", metavar="ID",
                      help="Record specific window by X11 ID")
    mode.add_argument("--select", action="store_true",
                      help="Draw rectangle to select area (like Shift+PrtScn)")
    mode.add_argument("--region", metavar="X,Y,W,H",
                      help="Record specific region coordinates")
    
    # Audio options
    parser.add_argument("--audio", "-a", action="store_true",
                        help="Record audio (desktop audio)")
    parser.add_argument("--audio-source", metavar="SRC",
                        help="PulseAudio source name")
    parser.add_argument("--list-audio", action="store_true",
                        help="List audio sources and exit")
    
    # Other options
    parser.add_argument("--bind-keys", action="store_true",
                        help="Show how to set up custom keyboard shortcuts")
    parser.add_argument("--list-monitors", action="store_true",
                        help="List monitors and exit")
    parser.add_argument("--list-windows", action="store_true",
                        help="List windows and exit")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show progress")
    parser.add_argument("-V", "--version", action="version",
                        version=f"%(prog)s {__version__}")
    
    args = parser.parse_args()
    
    # Handle --bind-keys first
    if args.bind_keys:
        print_keybinding_instructions()
        return 0
    
    # Handle list commands first (minimal deps)
    if args.list_audio:
        sources = list_audio_sources()
        if sources:
            print("Audio sources:")
            for name, desc in sources:
                print(f"  {name}")
        else:
            print("No audio sources found (PulseAudio may not be running)")
        return 0
    
    # Check if we need X11 interaction
    need_x11 = args.select or args.window or (
        not args.fullscreen and not args.monitor and
        not args.window_id and not args.region and
        not args.list_monitors and not args.list_windows
    )
    
    if not check_dependencies(need_x11):
        return 1
    
    # Handle other list commands
    if args.list_monitors:
        monitors = get_all_monitors()
        print(f"Found {len(monitors)} monitor(s):")
        for i, m in enumerate(monitors, 1):
            print(f"  {i}: {m['width']}x{m['height']} at ({m['left']}, {m['top']})")
        return 0
    
    if args.list_windows:
        windows = list_windows()
        if not windows:
            print("No windows found (install wmctrl)")
            return 1
        print(f"Found {len(windows)} window(s):")
        for wid, title in windows:
            print(f"  {wid}: {title}")
        return 0
    
    # Parse duration
    try:
        duration = parse_duration(args.duration)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    if duration <= 0:
        print("Error: Duration must be positive", file=sys.stderr)
        return 1
    
    if args.fps <= 0 or args.fps > 60:
        print("Error: FPS must be between 1 and 60", file=sys.stderr)
        return 1
    
    # Determine capture region
    region = None
    
    if args.fullscreen:
        region = get_full_screen()
        if args.verbose:
            print(f"Fullscreen: {region['width']}x{region['height']}", file=sys.stderr)
    
    elif args.monitor:
        monitors = get_all_monitors()
        if args.monitor < 1 or args.monitor > len(monitors):
            print(f"Error: Monitor {args.monitor} not found (1-{len(monitors)})",
                  file=sys.stderr)
            return 1
        region = monitors[args.monitor - 1]
    
    elif args.window:
        region = select_window_by_click(args.verbose)
        if not region:
            print("Error: Window selection failed", file=sys.stderr)
            return 1
    
    elif args.window_id:
        region = get_window_geometry(args.window_id)
        if not region:
            print(f"Error: Window {args.window_id} not found", file=sys.stderr)
            return 1
    
    elif args.region:
        try:
            parts = [int(x.strip()) for x in args.region.split(",")]
            if len(parts) != 4:
                raise ValueError()
            region = {"left": parts[0], "top": parts[1],
                      "width": parts[2], "height": parts[3]}
        except (ValueError, IndexError):
            print("Error: --region format: X,Y,WIDTH,HEIGHT", file=sys.stderr)
            return 1
    
    else:
        # Default: rectangle selection
        region = select_rectangle(args.verbose)
        if not region:
            print("Error: Selection too small (min 10x10)", file=sys.stderr)
            return 1
    
    # Validate
    if region["width"] < 10 or region["height"] < 10:
        print("Error: Region too small (min 10x10)", file=sys.stderr)
        return 1
    
    # Delay before recording starts
    countdown(args.delay, args.verbose)
    
    success = record_region(
        region, args.output, duration, args.fps,
        args.audio, args.audio_source, args.verbose
    )
    
    if success and os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"Saved: {args.output} ({size_mb:.1f} MB)")
        return 0
    else:
        print("Error: Recording failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
