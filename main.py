"""BeatSync SO-101: Expressive Audio-Driven Robotic Dancer.

Main execution entry point.
"""

import os
import sys
import argparse
import threading
from typing import Optional

from src.audio_processor import AudioBeatProcessor
from src.robot_controller import SO101Dancer


def play_audio_background(audio_path: str):
    """Plays audio in the background in sync with the robot dancer."""
    def _worker():
        try:
            if sys.platform == "win32" and audio_path.lower().endswith(".wav"):
                import winsound
                winsound.PlaySound(audio_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                # Cross-platform / MP3 fallback using soundfile + sounddevice or simple powershell media player
                if sys.platform == "win32":
                    import subprocess
                    ps_cmd = (
                        f'(New-Object Media.SoundPlayer "{os.path.abspath(audio_path)}").PlaySync()'
                        if audio_path.lower().endswith(".wav")
                        else f'Add-Type -AssemblyName presentationCore; $p = New-Object System.Windows.Media.MediaPlayer; $p.Open([System.Uri]"{os.path.abspath(audio_path)}"); $p.Play(); Start-Sleep -s 60'
                    )
                    subprocess.Popen(["powershell", "-c", ps_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            # Audio playback is optional, don't crash dance routine
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def main():
    parser = argparse.ArgumentParser(
        description="BeatSync SO-101: Expressive Audio-Driven Robotic Dancer for Hugging Face / Feetech SO-101"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to input audio file (WAV, MP3, FLAC, etc.)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate and run a punchy synth demo track with drops"
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port for Feetech motor bus (e.g. COM3 or /dev/ttyUSB0). Auto-detected if omitted."
    )
    parser.add_argument(
        "--sim",
        "--mock",
        action="store_true",
        dest="sim",
        help="Run in simulation mode with live ASCII HUD (no physical hardware required)"
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Disable audio speaker playback during dance"
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="Override detected BPM with fixed tempo"
    )
    parser.add_argument(
        "--offset-ms",
        type=float,
        default=70.0,
        help="Anticipatory latency compensation in ms so motion peaks hit on drum beats (default: 70)"
    )
    parser.add_argument(
        "--speed-limit",
        type=float,
        default=220.0,
        help="Maximum joint velocity limit in deg/s (default: 220)"
    )

    args = parser.parse_args()

    # Determine audio source
    audio_path = args.audio
    if not audio_path or args.demo:
        demo_file = os.path.join("tracks", "demo_beat.wav")
        if not os.path.exists(demo_file) or args.demo:
            print("[INFO] Generating synthetic electronic demo track (120 BPM, drops & build)...")
            audio_path = AudioBeatProcessor.generate_demo_track(demo_file, duration_sec=16.0, bpm=120.0)
        else:
            audio_path = demo_file

    print(f"🎵 Loading audio file: {audio_path}")
    processor = AudioBeatProcessor(audio_path)
    
    # Extract rich expressive beats
    detected_bpm, beat_events = processor.extract_expressive_beats()
    bpm = args.bpm if args.bpm else detected_bpm
    print(f"⚡ Detected BPM: {bpm:.1f} | Duration: {processor.duration:.2f}s | Total Beats: {len(beat_events)}")

    # Initialize Dancer with high-clarity trajectory engine
    dancer = SO101Dancer(
        port=args.port,
        force_sim=args.sim,
        speed_limit=args.speed_limit,
        offset_ms=args.offset_ms,
    )

    # Audio playback callback
    audio_callback = None
    if not args.no_sound:
        audio_callback = lambda: play_audio_background(audio_path)

    # Execute dance routine
    dancer.play_dance(
        beat_events=beat_events,
        bpm=bpm,
        duration=processor.duration,
        audio_playback_fn=audio_callback
    )


if __name__ == "__main__":
    main()
