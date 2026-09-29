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


def setup_audio_player(audio_path: str, volume: float = 0.8):
    """Sets up synchronized, low-latency background audio playback using sounddevice & soundfile."""
    try:
        import sounddevice as sd
        import soundfile as sf
        audio_data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        audio_data = audio_data * max(0.0, min(1.0, volume))

        def play():
            try:
                sd.play(audio_data, sr)
            except Exception as e:
                print(f"[Warning] Audio playback failed: {e}")

        def stop():
            try:
                sd.stop()
            except Exception:
                pass

        return play, stop
    except Exception as e:
        print(f"[Warning] sounddevice unavailable ({e}). Using system audio fallback.")
        def fallback_play():
            try:
                if sys.platform == "win32" and audio_path.lower().endswith(".wav"):
                    import winsound
                    winsound.PlaySound(audio_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception:
                pass
        return fallback_play, lambda: None


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
    parser.add_argument(
        "--volume",
        type=float,
        default=0.8,
        help="Playback volume level between 0.0 and 1.0 (default: 0.8)"
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

    # Low-latency synchronized audio player callbacks
    play_fn, stop_fn = None, None
    if not args.no_sound:
        print(f"🔊 Speaker playback enabled (Volume: {int(args.volume * 100)}%).")
        play_fn, stop_fn = setup_audio_player(audio_path, volume=args.volume)

    # Execute dance routine in sync with audio
    dancer.play_dance(
        beat_events=beat_events,
        bpm=bpm,
        duration=processor.duration,
        audio_playback_fn=play_fn,
        audio_stop_fn=stop_fn,
    )


if __name__ == "__main__":
    main()
