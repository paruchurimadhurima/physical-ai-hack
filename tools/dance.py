"""Make the SO-101 follower dance to a song using recorded move loops (moves/*.json).

Pipeline:
  1. Beat tracking (librosa) -> beat times; the bar start (downbeat) is guessed from onset strength.
  2. Choreography: a list of segments (start beat, move, tempo) - from the song's energy ("auto"),
     a fixed test sequence ("test"), or a JSON file from the team.
  3. Playback: the song's own beat grid drives the move phase, so the arm stays on the beat even if
     the tempo drifts. Moves are time-stretched (full tempo or half time), cross-faded over one beat
     at segment changes, and speed-limited per joint.

Safety: the arm first glides from its current pose to the first dance pose, and at the end (or on
Ctrl+C) glides back to where it started before the motors are released.

Usage:
    python tools/dance.py inputs/pdoom.wav --dry-run                    # plan + plot, no arm
    python tools/dance.py inputs/pdoom.wav --plan test --start 24 --duration 40
    python tools/dance.py inputs/pdoom.wav --offset-ms 80               # arm acts 80 ms earlier
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
MOVES_DIR = ROOT / "moves"
FOLLOWER_PORT = "/dev/tty.usbmodem5B8E1123681"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CONTROL_HZ = 50
FADE_BEATS = 1.0
GLIDE_S = 3.0
RATES = {"full": 1.0, "half": 0.5}


# ----------------------------------------------------------------------------- moves
@dataclass
class Move:
    name: str
    beats: int
    loop: np.ndarray  # (beats * ppb, 6)
    ppb: int

    @classmethod
    def load(cls, path: Path) -> "Move":
        d = json.loads(path.read_text())
        return cls(d["name"], int(d["beats"]), np.array(d["loop"], dtype=float), int(d["points_per_beat"]))

    def pose(self, phase_beats: float) -> np.ndarray:
        """Joint targets at a (fractional) position inside the loop, wrapping around."""
        x = (phase_beats % self.beats) * self.ppb
        i0 = int(np.floor(x)) % len(self.loop)
        i1 = (i0 + 1) % len(self.loop)
        f = x - np.floor(x)
        return (1 - f) * self.loop[i0] + f * self.loop[i1]

    def peak_speed(self, seconds_per_beat: float, rate: float) -> np.ndarray:
        """Peak joint speed (deg/s, gripper %/s) when played at `rate` move-beats per song-beat."""
        dt = seconds_per_beat / rate / self.ppb
        closed = np.vstack([self.loop, self.loop[:1]])
        return np.abs(np.diff(closed, axis=0)).max(axis=0) / dt


def load_moves() -> dict[str, Move]:
    moves = {p.stem: Move.load(p) for p in sorted(MOVES_DIR.glob("*.json"))}
    if not moves:
        raise SystemExit(f"Keine Moves in {MOVES_DIR} gefunden.")
    return moves


# ----------------------------------------------------------------------------- music
@dataclass
class Analysis:
    beats: np.ndarray  # seconds
    downbeat_phase: int  # index of the first downbeat within `beats`
    bar_energy: np.ndarray  # normalized 0..1, one value per bar starting at downbeat_phase
    tempo: float

    def beat_pos(self, t: float) -> float:
        """Continuous beat index at song time t (linear extrapolation outside the tracked range)."""
        b = self.beats
        if t < b[0]:
            return (t - b[0]) / np.median(np.diff(b[:8]))
        if t > b[-1]:
            return len(b) - 1 + (t - b[-1]) / np.median(np.diff(b[-8:]))
        return float(np.interp(t, b, np.arange(len(b))))

    def beat_time(self, beat_index: float) -> float:
        b = self.beats
        if beat_index <= 0:
            return b[0] + beat_index * np.median(np.diff(b[:8]))
        if beat_index >= len(b) - 1:
            return b[-1] + (beat_index - len(b) + 1) * np.median(np.diff(b[-8:]))
        return float(np.interp(beat_index, np.arange(len(b)), b))


def analyze(path: Path, bar_offset: int | None) -> Analysis:
    y, sr = librosa.load(path, sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    onset_t = librosa.times_like(onset, sr=sr)
    strength = np.interp(beats, onset_t, onset)
    phase = bar_offset if bar_offset is not None else int(np.argmax([strength[p::4].mean() for p in range(4)]))
    rms = librosa.feature.rms(y=y)[0]
    rms_t = librosa.times_like(rms, sr=sr)
    bar_starts = beats[phase::4]
    energy = np.array([rms[(rms_t >= a) & (rms_t < b)].mean() for a, b in zip(bar_starts[:-1], bar_starts[1:])])
    return Analysis(beats, phase, energy / energy.max(), float(np.atleast_1d(tempo)[0]))


# ----------------------------------------------------------------------------- choreography
@dataclass
class Segment:
    start_beat: float
    move: str
    tempo: str  # "full" | "half"
    label: str = ""


def choose_tempo(move: Move, spb: float, speed_limit: float, wanted: str) -> str:
    if wanted in RATES:
        return wanted
    return "full" if move.peak_speed(spb, 1.0).max() <= speed_limit else "half"


def plan_test(an: Analysis, moves, spb, speed_limit) -> list[Segment]:
    """Cycle through all moves, 4 bars each - for trying moves on the song."""
    segs, bar = [], 0
    names = list(moves)
    for k in range(len(an.bar_energy) // 4):
        name = names[k % len(names)]
        segs.append(Segment(an.downbeat_phase + 4 * bar, name, choose_tempo(moves[name], spb, speed_limit, "auto"), "test"))
        bar += 4
    return segs


def plan_auto(an: Analysis, moves, spb, speed_limit) -> list[Segment]:
    """Map the energy of each 4-bar phrase to a move family."""
    families = {
        "ruhig": [m for m in ("bounce",) if m in moves],
        "mittel": [m for m in ("sway", "snap") if m in moves],
        "stark": [m for m in ("bigwave", "snap") if m in moves],
    }
    for k, v in families.items():
        if not v:
            families[k] = list(moves)
    segs, counters = [], {k: 0 for k in families}
    for p in range(0, len(an.bar_energy), 4):
        e = an.bar_energy[p : p + 4].mean()
        level = "ruhig" if e < 0.55 else "mittel" if e < 0.85 else "stark"
        name = families[level][counters[level] % len(families[level])]
        counters[level] += 1
        start = an.downbeat_phase + 4 * p
        if segs and segs[-1].move == name and segs[-1].label == level:
            continue
        segs.append(Segment(start, name, choose_tempo(moves[name], spb, speed_limit, "auto"), level))
    return segs


def plan_from_file(path: Path, an: Analysis, moves, spb, speed_limit) -> list[Segment]:
    """JSON from the team: [{"start": seconds, "move": "bounce", "tempo": "half"|"full"|"auto"}, ...].

    Each start time is snapped to the nearest bar start so moves change on the "one".
    """
    segs = []
    for item in json.loads(path.read_text()):
        if item["move"] not in moves:
            raise SystemExit(f"Move '{item['move']}' gibt es nicht. Vorhanden: {', '.join(moves)}")
        b = an.beat_pos(float(item["start"]))
        bar_b = an.downbeat_phase + 4 * round((b - an.downbeat_phase) / 4)
        tempo = choose_tempo(moves[item["move"]], spb, speed_limit, item.get("tempo", "auto"))
        segs.append(Segment(bar_b, item["move"], tempo, item.get("label", "")))
    return sorted(segs, key=lambda s: s.start_beat)


def smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def target_pose(bp: float, segs: list[Segment], moves: dict[str, Move]) -> np.ndarray:
    k = max(0, int(np.searchsorted([s.start_beat for s in segs], bp, side="right")) - 1)
    s = segs[k]
    pose = moves[s.move].pose(max(0.0, bp - s.start_beat) * RATES[s.tempo])
    if k > 0 and bp - s.start_beat < FADE_BEATS:
        p = segs[k - 1]
        prev = moves[p.move].pose((bp - p.start_beat) * RATES[p.tempo])
        w = smoothstep((bp - s.start_beat) / FADE_BEATS)
        pose = (1 - w) * prev + w * pose
    return pose


# ----------------------------------------------------------------------------- playback
def glide(robot, start: np.ndarray, end: np.ndarray, seconds: float):
    n = int(seconds * CONTROL_HZ)
    for i in range(1, n + 1):
        t0 = time.perf_counter()
        q = start + smoothstep(i / n) * (end - start)
        robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, q)})
        time.sleep(max(0.0, 1 / CONTROL_HZ - (time.perf_counter() - t0)))


def simulate(an, segs, moves, t_start, t_end, offset, speed_limit):
    """Offline run of the exact control loop; returns times, commanded poses and clip counts."""
    ts = np.arange(t_start, t_end, 1 / CONTROL_HZ)
    max_step = speed_limit / CONTROL_HZ
    q_prev = target_pose(an.beat_pos(t_start + offset), segs, moves)
    out, clipped = [], np.zeros(len(JOINTS), dtype=int)
    for t in ts:
        want = target_pose(an.beat_pos(t + offset), segs, moves)
        q = q_prev + np.clip(want - q_prev, -max_step, max_step)
        clipped += np.abs(want - q) > 1.0
        out.append(q)
        q_prev = q
    return ts, np.array(out), clipped


def save_plot(path, an, segs, ts, qs, t_start, t_end):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(JOINTS) + 1, 1, figsize=(14, 12), sharex=True)
    bar_t = an.beats[an.downbeat_phase :: 4][: len(an.bar_energy)]
    axes[0].step(bar_t, an.bar_energy, where="post", color="0.3")
    axes[0].set_ylabel("Energie", fontsize=8)
    colors = {}
    for k, s in enumerate(segs):
        a = an.beat_time(s.start_beat)
        b = an.beat_time(segs[k + 1].start_beat) if k + 1 < len(segs) else t_end
        c = colors.setdefault(s.move, f"C{len(colors)}")
        for ax in axes:
            ax.axvspan(a, b, color=c, alpha=0.12)
        axes[0].text(a, 1.02, f"{s.move} ({s.tempo})", fontsize=7, rotation=30)
    for j, ax in enumerate(axes[1:]):
        ax.plot(ts, qs[:, j], lw=0.8)
        ax.set_ylabel(JOINTS[j], fontsize=8)
    axes[-1].set_xlim(t_start, t_end)
    axes[-1].set_xlabel("Songzeit (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("song", type=Path)
    p.add_argument("--plan", default="auto", help="'auto' (Energie), 'test' (alle Moves nacheinander) oder Pfad zu JSON")
    p.add_argument("--start", type=float, default=0.0, help="Startzeit im Song (s)")
    p.add_argument("--duration", type=float, default=None, help="Laenge des Ausschnitts (s), Standard: bis zum Ende")
    p.add_argument("--offset-ms", type=float, default=0.0, help="Arm reagiert so viele ms frueher (Latenzausgleich)")
    p.add_argument("--speed-limit", type=float, default=200.0, help="Max. Gelenktempo in Grad/s (Standard 200)")
    p.add_argument("--bar-offset", type=int, default=None, help="Taktanfang manuell setzen (0-3), falls falsch erkannt")
    p.add_argument("--volume", type=float, default=0.8)
    p.add_argument("--dry-run", action="store_true", help="Nur planen, simulieren und plotten - Arm bleibt aus")
    args = p.parse_args()

    moves = load_moves()
    print(f"Moves: {', '.join(f'{m.name} ({m.beats} Beats)' for m in moves.values())}")
    print("Analysiere Song ...")
    an = analyze(args.song, args.bar_offset)
    spb = float(np.median(np.diff(an.beats)))
    print(f"  {an.tempo:.1f} BPM | {len(an.beats)} Beats | erster Taktanfang bei {an.beats[an.downbeat_phase]:.2f}s "
          f"(Beat-Phase {an.downbeat_phase}) | {len(an.bar_energy)} Takte")

    if args.plan == "auto":
        segs = plan_auto(an, moves, spb, args.speed_limit)
    elif args.plan == "test":
        segs = plan_test(an, moves, spb, args.speed_limit)
    else:
        segs = plan_from_file(Path(args.plan), an, moves, spb, args.speed_limit)

    audio, sr = sf.read(args.song, dtype="float32", always_2d=True)
    song_len = len(audio) / sr
    t_start = args.start
    t_end = min(song_len, t_start + args.duration) if args.duration else song_len
    offset = args.offset_ms / 1000.0

    print("\nChoreografie:")
    for k, s in enumerate(segs):
        a = an.beat_time(s.start_beat)
        b = an.beat_time(segs[k + 1].start_beat) if k + 1 < len(segs) else song_len
        if b < t_start or a > t_end:
            continue
        v = moves[s.move].peak_speed(spb, RATES[s.tempo]).max()
        warn = "  <-- schneller als Limit, wird gebremst" if v > args.speed_limit else ""
        print(f"  {a:6.1f}s - {b:6.1f}s  {s.move:8s} {s.tempo:4s}  {s.label:7s} Spitze {v:4.0f} Grad/s{warn}")

    ts, qs, clipped = simulate(an, segs, moves, t_start, t_end, offset, args.speed_limit)
    plot = ROOT / "outputs" / f"dance_{args.song.stem}.png"
    plot.parent.mkdir(exist_ok=True)
    save_plot(plot, an, segs, ts, qs, t_start, t_end)
    print(f"\nSimulation: {len(ts)} Steuerschritte | Tempo-Limit griff (>1 Grad) pro Gelenk: {dict(zip(JOINTS, clipped.tolist()))}")
    print(f"Plot: {plot}")
    if args.dry_run:
        return

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="follower"))
    robot.connect()
    obs = robot.get_observation()
    home = np.array([obs[f"{j}.pos"] for j in JOINTS])
    q_prev = home.copy()
    try:
        first = target_pose(an.beat_pos(t_start + offset), segs, moves)
        print(f"\nGleite in {GLIDE_S:.0f}s zur ersten Tanzpose ...")
        glide(robot, home, first, GLIDE_S)
        q_prev = first
        chunk = audio[int(t_start * sr) : int(t_end * sr)] * args.volume
        sd.play(chunk, sr)
        t0 = time.perf_counter() + sd.get_stream().latency
        max_step = args.speed_limit / CONTROL_HZ
        print("Musik! (Ctrl+C zum Stoppen)")
        while True:
            tick = time.perf_counter()
            t_song = t_start + (tick - t0)
            if t_song >= t_end:
                break
            want = target_pose(an.beat_pos(t_song + offset), segs, moves)
            q = q_prev + np.clip(want - q_prev, -max_step, max_step)
            robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, q)})
            q_prev = q
            time.sleep(max(0.0, 1 / CONTROL_HZ - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        print("\nGestoppt.")
    finally:
        sd.stop()
        print(f"Gleite in {GLIDE_S:.0f}s zurueck zur Startpose ...")
        glide(robot, q_prev, home, GLIDE_S)
        robot.disconnect()


if __name__ == "__main__":
    main()
