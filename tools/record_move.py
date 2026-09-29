"""Record a dance move from the leader arm as a seamless, beat-aligned loop.

Runs normal teleoperation (leader -> follower) the whole time, plays a metronome,
and records `--loops` repetitions of a `--beats`-long move after a one-bar count-in.
The repetitions are phase-aligned, averaged (smooths out hand jitter) and the loop is
closed so its end matches its start. Result: moves/<name>.json (+ a plot as .png).

Tips:
  * Start and end every move in the same neutral dance pose.
  * Let the turning points (direction changes) land exactly on the clicks.
  * Record slower than the song (e.g. 90 BPM); playback time-stretches the move.

Usage:
    python tools/record_move.py wave --bpm 90 --beats 4 --loops 4
"""

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.ndimage import gaussian_filter1d
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

FOLLOWER_PORT = "/dev/tty.usbmodem5B8E1123681"
LEADER_PORT = "/dev/tty.usbmodem5B8E1150551"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CONTROL_HZ = 50
POINTS_PER_BEAT = 32
SONG_BPM = 129  # p(doom); used only to report how fast the move would be at song tempo
SR = 44100
MOVES_DIR = Path(__file__).resolve().parent.parent / "moves"


def click_track(bpm: float, n_beats: int, beats_per_bar: int) -> np.ndarray:
    """Metronome audio: high click on beat 1 of each bar, low click otherwise."""
    beat_len = 60.0 / bpm
    audio = np.zeros(int(SR * beat_len * (n_beats + 1)), dtype=np.float32)
    t = np.arange(int(0.03 * SR)) / SR
    env = np.exp(-t * 120)
    for b in range(n_beats):
        freq = 1600 if b % beats_per_bar == 0 else 900
        start = int(b * beat_len * SR)
        audio[start : start + len(t)] += 0.6 * env * np.sin(2 * np.pi * freq * t)
    return audio


class Teleop(threading.Thread):
    """Leader -> follower at CONTROL_HZ; records (time, leader action) while `recording` is set."""

    def __init__(self, leader, follower):
        super().__init__(daemon=True)
        self.leader, self.follower = leader, follower
        self.recording = threading.Event()
        self.stop_flag = threading.Event()
        self.samples: list[tuple[float, list[float]]] = []
        self.error: Exception | None = None

    def run(self):
        period = 1.0 / CONTROL_HZ
        try:
            while not self.stop_flag.is_set():
                t0 = time.perf_counter()
                action = self.leader.get_action()
                self.follower.send_action(action)
                if self.recording.is_set():
                    self.samples.append((t0, [float(action[f"{j}.pos"]) for j in JOINTS]))
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
        except Exception as e:  # surfaced in the main thread
            self.error = e


MAX_SPREAD_FOR_MEAN = 8.0  # degrees; above this the repetitions disagree and averaging smears the move


def cut_repetitions(samples, t_first_beat, beat_len, beats, loops):
    t = np.array([s[0] for s in samples])
    q = np.array([s[1] for s in samples])
    beat_pos = (t - t_first_beat) / beat_len
    grid = np.arange(beats * POINTS_PER_BEAT) / POINTS_PER_BEAT
    reps = []
    for k in range(loops):
        target = k * beats + grid
        if target[0] < beat_pos[0] or target[-1] > beat_pos[-1]:
            continue
        reps.append(np.stack([np.interp(target, beat_pos, q[:, j]) for j in range(len(JOINTS))], axis=1))
    return np.array(reps)


def align_to_beats(loop):
    """Rotate the loop so the turning points of its biggest joint land on the beat grid.

    People dancing along to a click are consistently a little late; because the loop is
    cyclic this shift is lossless. Returns (aligned_loop, shift_in_beats).
    """
    j = int(np.argmax(loop.max(0) - loop.min(0)))
    x = loop[:, j]
    rng = x.max() - x.min()
    n = len(x)
    turns = [
        k for k in range(n)
        if (x[k] - x[k - 1]) * (x[(k + 1) % n] - x[k]) < 0 and abs(x[k] - x.mean()) > 0.2 * rng
    ]
    if not turns or rng < 3:
        return loop, 0.0
    phases = np.array(turns) / POINTS_PER_BEAT
    # Circular mean of the fractional offsets to the nearest beat.
    ang = 2 * np.pi * (phases % 1.0)
    shift = float(np.angle(np.exp(1j * ang).mean()) / (2 * np.pi))  # in (-0.5, 0.5]
    return np.roll(loop, -int(round(shift * POINTS_PER_BEAT)), axis=0), shift


def build_loop(samples, t_first_beat, beat_len, beats, loops, skip_first, smooth_beats=0.125, align=True):
    reps = cut_repetitions(samples, t_first_beat, beat_len, beats, loops)
    return loop_from_reps(reps, beats, skip_first, smooth_beats, align)


def loop_from_reps(reps, beats, skip_first=True, smooth_beats=0.125, align=True):
    grid = np.arange(beats * POINTS_PER_BEAT) / POINTS_PER_BEAT
    used = reps[1:] if skip_first and len(reps) >= 3 else reps
    spread = used.std(axis=0).mean(axis=0)
    if spread.max() <= MAX_SPREAD_FOR_MEAN:
        method = "mittelwert"
        loop = used.mean(axis=0)
    else:
        # Repetitions disagree: averaging would smear them, so take the most typical one.
        dist = [np.abs(r - used.mean(axis=0)).mean() for r in used]
        method = f"einzelne Wiederholung #{int(np.argmin(dist)) + 1} (zu uneinheitlich zum Mitteln)"
        loop = used[int(np.argmin(dist))].copy()
    # Close the loop: bend only the last part of the move (smoothstep over `fade` beats) so its end
    # flows into its start. A ramp over the whole loop would shift the entire second half of the move.
    end_next = 2 * loop[-1] - loop[-2]
    drift = end_next - loop[0]
    fade = max(1.0, beats / 4)
    u = np.clip((grid - (beats - fade)) / fade, 0.0, 1.0)
    loop = loop - np.outer(u * u * (3 - 2 * u), drift)
    # Remove leftover hand jitter; mode="wrap" keeps the loop seamless.
    loop = gaussian_filter1d(loop, sigma=smooth_beats * POINTS_PER_BEAT, axis=0, mode="wrap")
    # Safety: never command a pose outside the range the arm actually visited while recording.
    loop = np.clip(loop, used.min(axis=(0, 1)), used.max(axis=(0, 1)))
    shift = 0.0
    if align:
        loop, shift = align_to_beats(loop)
    return loop, reps, used, spread, drift, method, shift


def peak_speed(loop, bpm):
    """Peak joint speed in degrees (gripper: %) per second when played at `bpm`."""
    dt = 60.0 / bpm / POINTS_PER_BEAT
    closed = np.vstack([loop, loop[:1]])
    return np.abs(np.diff(closed, axis=0)).max(axis=0) / dt


def save_plot(path, reps, loop, beats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.arange(loop.shape[0]) / POINTS_PER_BEAT
    fig, axes = plt.subplots(len(JOINTS), 1, figsize=(8, 11), sharex=True)
    for j, ax in enumerate(axes):
        for r in reps:
            ax.plot(grid, r[:, j], color="0.75", lw=1)
        ax.plot(grid, loop[:, j], color="C0", lw=2)
        for b in range(beats + 1):
            ax.axvline(b, color="C3", lw=0.6, alpha=0.5)
        ax.set_ylabel(JOINTS[j], fontsize=8)
    axes[-1].set_xlabel("Beat (grau = einzelne Wiederholungen, blau = gemittelter Loop)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="Name of the move, e.g. 'wave'")
    p.add_argument("--bpm", type=float, default=90, help="Metronome tempo while recording (default 90)")
    p.add_argument("--beats", type=int, default=4, help="Length of one loop in beats (default 4)")
    p.add_argument("--loops", type=int, default=4, help="Repetitions to record and average (default 4)")
    p.add_argument("--keep-first", action="store_true", help="Also average the first repetition")
    p.add_argument("--smooth", type=float, default=0.125, help="Smoothing width in beats (default 1/8 beat)")
    args = p.parse_args()

    beat_len = 60.0 / args.bpm
    count_in = 4
    n_beats = count_in + args.beats * args.loops

    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id="leader"))
    follower = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="follower"))
    leader.connect()
    follower.connect()
    teleop = Teleop(leader, follower)
    teleop.start()
    try:
        print(f"\nTeleoperation laeuft. Bring den Arm in deine neutrale Tanzpose.")
        input(f"Enter = 4 Takte Einzaehlen, dann {args.loops}x {args.beats} Beats '{args.name}' bei {args.bpm:.0f} BPM ...")
        audio = click_track(args.bpm, n_beats, beats_per_bar=4)
        sd.play(audio, SR)
        t_audio = time.perf_counter() + sd.get_stream().latency
        t_first_beat = t_audio + count_in * beat_len
        time.sleep(max(0.0, t_first_beat - time.perf_counter() - 0.5))
        teleop.recording.set()
        print("  ... 3, 2, 1 - los!")
        time.sleep(max(0.0, t_first_beat + args.beats * args.loops * beat_len + 0.3 - time.perf_counter()))
        teleop.recording.clear()
        sd.wait()
        if teleop.error:
            raise teleop.error

        loop, reps, used, spread, drift, method, shift = build_loop(
            teleop.samples, t_first_beat, beat_len, args.beats, args.loops, not args.keep_first, args.smooth
        )
        MOVES_DIR.mkdir(exist_ok=True)
        out = MOVES_DIR / f"{args.name}.json"
        out.write_text(
            json.dumps(
                {
                    "name": args.name,
                    "recorded_bpm": args.bpm,
                    "beats": args.beats,
                    "points_per_beat": POINTS_PER_BEAT,
                    "joints": JOINTS,
                    "loop": np.round(loop, 2).tolist(),
                    "spread_deg": np.round(spread, 2).tolist(),
                    "repetitions_used": int(len(used)),
                    "method": method,
                    "beat_shift": round(shift, 3),
                    "repetitions": np.round(reps, 2).tolist(),
                },
                indent=1,
            )
        )
        save_plot(out.with_suffix(".png"), reps, loop, args.beats)

        print(f"\nGespeichert: {out}  (+ Plot {out.with_suffix('.png').name})")
        print(f"Verwendet: {method} ({len(used)} von {len(reps)} Wiederholungen)")
        print(f"Auf den Beat verschoben um {shift:+.2f} Beats (du warst {'spaet' if shift > 0 else 'frueh'} dran)")
        print("Gelenk          Streuung   Loop-Korrektur   Spitzentempo @ " f"{args.bpm:.0f} / {SONG_BPM} / {SONG_BPM / 2:.1f} BPM")
        v_rec, v_song, v_half = (peak_speed(loop, b) for b in (args.bpm, SONG_BPM, SONG_BPM / 2))
        for j, name in enumerate(JOINTS):
            print(
                f"  {name:14s} {spread[j]:5.1f}      {drift[j]:+6.1f}          "
                f"{v_rec[j]:5.0f} / {v_song[j]:5.0f} / {v_half[j]:5.0f} Grad/s"
            )
        print("\nStreuung > ~5 Grad: Wiederholungen waren uneinheitlich -> neu aufnehmen.")
        print("Loop-Korrektur gross: Endpose != Startpose -> sauberer in die Neutralpose zurueckkehren.")
        input("\nLeader zurueck in die Ruheposition fuehren, dann Enter zum Beenden ...")
    finally:
        teleop.stop_flag.set()
        teleop.join(timeout=2)
        leader.disconnect()
        follower.disconnect()


if __name__ == "__main__":
    main()
