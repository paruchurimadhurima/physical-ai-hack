"""Audio beat extraction and musical feature analysis for BeatSync SO-101."""

import os
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import librosa
import soundfile as sf


@dataclass
class BeatEvent:
    time: float          # Timestamp in seconds
    bpm: float           # Local or global BPM
    energy: float        # Normalized energy (0.0 - 1.0)
    is_downbeat: bool    # True if downbeat (bar start / kick)
    is_drop: bool        # True if energetic drop / peak accent
    section: str         # "intro", "groove", "buildup", "drop"


class AudioBeatProcessor:
    """Extracts beat timestamps, onset energy, and musical dynamics from audio."""

    def __init__(self, audio_path: str):
        self.audio_path = audio_path
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio (mono)
        self.y, self.sr = librosa.load(audio_path, sr=None)
        self.duration = float(librosa.get_duration(y=self.y, sr=self.sr))

    def extract_beats(self) -> Tuple[float, List[float]]:
        """Basic extraction: returns tempo (BPM) and list of beat timestamps."""
        tempo, beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        bpm = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)
        return bpm, [float(t) for t in beat_times]

    def extract_expressive_beats(self) -> Tuple[float, List[BeatEvent]]:
        """Advanced extraction: returns BPM and detailed musical beat events with dynamics."""
        bpm, raw_beat_times = self.extract_beats()
        
        # Compute onset strength envelope
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        times = librosa.times_like(onset_env, sr=self.sr)

        # Normalize onset envelope
        max_onset = np.max(onset_env) if len(onset_env) > 0 and np.max(onset_env) > 0 else 1.0
        norm_onset = onset_env / max_onset

        # Compute spectral centroid to distinguish bass drops from high hats
        spec_cent = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        max_cent = np.max(spec_cent) if len(spec_cent) > 0 and np.max(spec_cent) > 0 else 1.0
        norm_cent = spec_cent / max_cent

        # Map each beat to a rich BeatEvent
        events: List[BeatEvent] = []
        for i, b_time in enumerate(raw_beat_times):
            # Find nearest frame in onset envelope
            frame_idx = int(np.clip(librosa.time_to_frames(b_time, sr=self.sr), 0, len(norm_onset) - 1))
            energy = float(norm_onset[frame_idx])
            
            # Simple bar estimation: every 4th beat is downbeat in 4/4 time
            is_downbeat = (i % 4 == 0)
            
            # Detect drop: high energy beat preceded by lower energy
            prev_energy = norm_onset[max(0, frame_idx - 15)]
            is_drop = bool(energy > 0.65 and (energy - prev_energy) > 0.25)

            # Classify section
            progress = b_time / self.duration if self.duration > 0 else 0
            if progress < 0.15:
                section = "intro"
            elif is_drop or energy > 0.7:
                section = "drop"
            elif energy > 0.4:
                section = "groove"
            else:
                section = "chill"

            events.append(BeatEvent(
                time=float(b_time),
                bpm=bpm,
                energy=energy,
                is_downbeat=is_downbeat,
                is_drop=is_drop,
                section=section
            ))

        return bpm, events

    @staticmethod
    def generate_demo_track(output_path: str = "tracks/demo_beat.wav", duration_sec: float = 16.0, bpm: float = 120.0) -> str:
        """Generates a punchy synth kick/snare/hi-hat demo track with bass drops for instant testing."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sr = 44100
        total_samples = int(duration_sec * sr)
        t = np.linspace(0, duration_sec, total_samples, endpoint=False)
        audio = np.zeros(total_samples)

        beat_interval = 60.0 / bpm
        num_beats = int(duration_sec / beat_interval)

        # Kick drum synthesis function
        def synth_kick(length_samples):
            kt = np.linspace(0, 0.2, length_samples)
            freq = 150 * np.exp(-30 * kt) + 45
            envelope = np.exp(-12 * kt)
            return 0.8 * np.sin(2 * np.pi * freq * kt) * envelope

        # Snare drum synthesis function
        def synth_snare(length_samples):
            st = np.linspace(0, 0.15, length_samples)
            noise = np.random.uniform(-1, 1, length_samples)
            tone = np.sin(2 * np.pi * 200 * st)
            envelope = np.exp(-18 * st)
            return 0.5 * (noise * 0.7 + tone * 0.3) * envelope

        # Hi-hat synthesis function
        def synth_hihat(length_samples):
            ht = np.linspace(0, 0.05, length_samples)
            noise = np.random.uniform(-1, 1, length_samples)
            envelope = np.exp(-40 * ht)
            return 0.25 * noise * envelope

        for b in range(num_beats):
            beat_time = b * beat_interval
            start_idx = int(beat_time * sr)

            # Kick on beats 1 and 3 (or all 4 during drop sections)
            if b % 2 == 0 or (b >= num_beats // 2 and b < 3 * num_beats // 4):
                kick_len = min(int(0.2 * sr), total_samples - start_idx)
                if kick_len > 0:
                    audio[start_idx:start_idx + kick_len] += synth_kick(kick_len)

            # Snare on beats 2 and 4
            if b % 2 == 1:
                snare_len = min(int(0.15 * sr), total_samples - start_idx)
                if snare_len > 0:
                    audio[start_idx:start_idx + snare_len] += synth_snare(snare_len)

            # 8th note hi-hats
            hat_len = min(int(0.05 * sr), total_samples - start_idx)
            if hat_len > 0:
                audio[start_idx:start_idx + hat_len] += synth_hihat(hat_len)

            # Offbeat hi-hat
            offbeat_start = int((beat_time + beat_interval / 2) * sr)
            offbeat_len = min(int(0.05 * sr), total_samples - offbeat_start)
            if offbeat_start < total_samples and offbeat_len > 0:
                audio[offbeat_start:offbeat_start + offbeat_len] += synth_hihat(offbeat_len)

        # Normalize audio
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9

        sf.write(output_path, audio, sr)
        return output_path
