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

    @staticmethod
    def generate_performance_track(
        output_path: str = "tracks/demo_performance_2min.wav",
        duration_sec: float = 120.0,
        bpm: float = 126.0,
    ) -> str:
        """Generates a full 2-minute multi-section electronic showcase track with chill, build-up, and drop sections."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sr = 44100
        total_samples = int(duration_sec * sr)
        audio = np.zeros(total_samples, dtype=np.float32)
        spb = 60.0 / bpm
        total_beats = int(duration_sec / spb)

        def add_kick(start_idx, punch=1.0, sub_heavy=False):
            k_len = min(int(0.24 * sr), total_samples - start_idx)
            if k_len <= 0: return
            t = np.linspace(0, k_len / sr, k_len)
            freq_start = 180 if punch > 1.0 else 150
            freq_decay = 40 if sub_heavy else 30
            freq = freq_start * np.exp(-freq_decay * t) + (42 if sub_heavy else 50)
            env = np.exp(-12 * t)
            click = 0.3 * np.exp(-120 * t) * np.sin(2 * np.pi * 1200 * t)
            audio[start_idx:start_idx + k_len] += (0.85 * np.sin(2 * np.pi * freq * t) + click) * env * punch

        def add_snare(start_idx, snappy=1.0):
            s_len = min(int(0.18 * sr), total_samples - start_idx)
            if s_len <= 0: return
            t = np.linspace(0, s_len / sr, s_len)
            noise = np.random.uniform(-1, 1, s_len)
            tone = np.sin(2 * np.pi * 210 * t) * np.exp(-25 * t)
            env = np.exp(-18 * t)
            audio[start_idx:start_idx + s_len] += (0.65 * noise + 0.35 * tone) * env * snappy

        def add_hihat(start_idx, vol=0.25, open_hat=False):
            h_len = min(int((0.15 if open_hat else 0.04) * sr), total_samples - start_idx)
            if h_len <= 0: return
            t = np.linspace(0, h_len / sr, h_len)
            noise = np.random.uniform(-1, 1, h_len)
            env = np.exp((-15 if open_hat else -55) * t)
            audio[start_idx:start_idx + h_len] += noise * env * vol

        def add_synth_tone(start_idx, freq, length_sec, vol=0.3, saw_harmonics=False):
            n_len = min(int(length_sec * sr), total_samples - start_idx)
            if n_len <= 0: return
            t = np.linspace(0, length_sec, n_len)
            tone = np.sin(2 * np.pi * freq * t)
            if saw_harmonics:
                tone += 0.4 * np.sin(2 * np.pi * freq * 2 * t) + 0.2 * np.sin(2 * np.pi * freq * 3 * t)
            env = np.exp(-3.0 * t)
            audio[start_idx:start_idx + n_len] += tone * env * vol

        chords = [
            [220.0, 261.6, 329.6, 440.0],  # Am
            [174.6, 220.0, 261.6, 349.2],  # F
            [130.8, 164.8, 196.0, 261.6],  # C
            [196.0, 246.9, 293.7, 392.0]   # G
        ]
        bass_notes = [55.0, 43.65, 65.4, 49.0]

        for b in range(total_beats):
            bt = b * spb
            idx = int(bt * sr)
            bar = b // 4
            bar_beat = b % 4
            chord = chords[(bar // 2) % len(chords)]
            root = bass_notes[(bar // 2) % len(bass_notes)]

            # 1. Intro (Bars 0-7, 0-15s): Chill low beat sway
            if bar < 8:
                add_synth_tone(idx, chord[bar_beat], spb * 1.5, vol=0.35)
                if bar >= 4 and bar_beat in (1, 3):
                    add_hihat(idx, vol=0.15)
            # 2. Groove Introduction (Bars 8-15, 15-30s): Upbeat bounce starts
            elif bar < 16:
                if bar_beat in (0, 2): add_kick(idx, punch=0.9)
                if bar_beat in (1, 3): add_snare(idx, snappy=0.7)
                add_hihat(idx + int(spb * 0.5 * sr), vol=0.2)
                add_synth_tone(idx, root, spb * 0.8, vol=0.3, saw_harmonics=True)
                add_synth_tone(idx, chord[bar_beat], spb * 0.8, vol=0.25)
            # 3. High-Energy Groove 1 (Bars 16-27, 30-53s): Driving rhythm snaps
            elif bar < 28:
                add_kick(idx, punch=1.0)
                if bar_beat in (1, 3): add_snare(idx, snappy=1.0)
                for s in range(4):
                    add_hihat(idx + int(s * spb / 4 * sr), vol=(0.28 if s == 2 else 0.14))
                add_synth_tone(idx, root, spb * 0.5, vol=0.4, saw_harmonics=True)
                add_synth_tone(idx + int(spb * 0.5 * sr), root * 1.5, spb * 0.4, vol=0.3, saw_harmonics=True)
            # 4. Climactic Drop 1 (Bars 28-35, 53-68s): Massive bass wave
            elif bar < 36:
                add_kick(idx, punch=1.2, sub_heavy=True)
                if bar_beat in (1, 3):
                    add_snare(idx, snappy=1.1)
                    add_hihat(idx, vol=0.35, open_hat=True)
                for s in range(4): add_hihat(idx + int(s * spb / 4 * sr), vol=0.2)
                add_synth_tone(idx, root / 1.5, spb * 0.9, vol=0.5, saw_harmonics=True)
                add_synth_tone(idx, chord[bar_beat] * 1.5, spb * 0.6, vol=0.35, saw_harmonics=True)
            # 5. Flowy Ambient Breakdown (Bars 36-43, 68-83s): Flowing piano/strings
            elif bar < 44:
                add_synth_tone(idx, chord[bar_beat], spb * 2.0, vol=0.45)
                if bar_beat == 0: add_synth_tone(idx, root * 2, spb * 3.5, vol=0.35)
            # 6. Snare Roll Build-up (Bars 44-47, 83-91s): Rising excitement
            elif bar < 48:
                progress = (b - 44 * 4) / 16.0
                sub_count = 2 if progress < 0.4 else (4 if progress < 0.8 else 8)
                for s in range(sub_count):
                    add_snare(idx + int(s * spb / sub_count * sr), snappy=0.3 + 0.7 * progress)
                riser_f = 150 + 800 * progress
                add_synth_tone(idx, riser_f, spb, vol=0.2 + 0.3 * progress, saw_harmonics=True)
            # 7. Ultimate Climax Drop 2 (Bars 48-59, 91-114s): Peak dance performance!
            elif bar < 60:
                add_kick(idx, punch=1.25, sub_heavy=True)
                add_snare(idx if bar_beat in (1, 3) else idx + int(spb * 0.5 * sr), snappy=1.0)
                for s in range(4): add_hihat(idx + int(s * spb / 4 * sr), vol=0.25)
                if bar_beat == 2: add_hihat(idx, vol=0.35, open_hat=True)
                add_synth_tone(idx, root, spb * 0.8, vol=0.45, saw_harmonics=True)
                add_synth_tone(idx, chord[(bar_beat + 1) % 4] * 2, spb * 0.5, vol=0.35, saw_harmonics=True)
            # 8. Outro / Safe Cooldown (Bars 60-63, 114-120s)
            else:
                fade = max(0.0, 1.0 - (b - 60 * 4) / 16.0)
                add_synth_tone(idx, chord[bar_beat], spb * 1.5, vol=0.35 * fade)

        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            audio = (audio / max_amp) * 0.94

        sf.write(output_path, audio, sr)
        return output_path

