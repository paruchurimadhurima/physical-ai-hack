"""High-clarity robot controller and choreography engine for BeatSync SO-101.

Features:
- Continuous 32-point-per-beat phase-locked trajectory playback from recorded move loops (moves/*.json)
- Anticipatory latency compensation (--offset-ms, default 70ms) to hit physical turning points on acoustic transients
- Dynamic segment planning based on musical energy, downbeats, and drops
- 1-beat smoothstep crossfading between dance moves
- Sharp micro-accents on drum transients (downbeats and snare claps)
- Cross-platform hardware bus (LeRobotFeetechBus) with COM port discovery & in-memory simulation (MockRobotBus)
- Safe glide-in from home position and glide-out to safe neutral on exit
"""

import sys
import os
import time
import math
import json
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from src.audio_processor import BeatEvent
from src.visualizer import DanceVisualizer


# Standard joint names matching SO-101 / SO-ARM100
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CONTROL_HZ = 50  # 50 Hz matches recorded motion resolution & bus throughput
FADE_BEATS = 1.0
GLIDE_S = 1.5


def smoothstep(u: float) -> float:
    """Hermite interpolation smoothstep for seamless crossfading."""
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


@dataclass
class MoveLoop:
    """Continuous 6-DOF dance trajectory loop with sub-beat interpolation."""
    name: str
    beats: int
    loop: np.ndarray  # shape (beats * ppb, 6)
    ppb: int

    @classmethod
    def load(cls, path: str) -> "MoveLoop":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            name=d["name"],
            beats=int(d["beats"]),
            loop=np.array(d["loop"], dtype=float),
            ppb=int(d.get("points_per_beat", 32)),
        )

    def pose(self, phase_beats: float) -> np.ndarray:
        """Evaluates 6-DOF joint targets at fractional beat phase with sub-beat linear interpolation."""
        x = (phase_beats % self.beats) * self.ppb
        i0 = int(np.floor(x)) % len(self.loop)
        i1 = (i0 + 1) % len(self.loop)
        f = x - np.floor(x)
        return (1.0 - f) * self.loop[i0] + f * self.loop[i1]

    @classmethod
    def create_synthetic(
        cls,
        name: str,
        beats: int,
        pose_a: Dict[str, float],
        pose_b: Dict[str, float],
        ppb: int = 32,
    ) -> "MoveLoop":
        """Creates smooth harmonic fallback loop if no recorded json file exists."""
        n_points = beats * ppb
        arr_a = np.array([pose_a.get(j, 0.0) for j in JOINTS], dtype=float)
        arr_b = np.array([pose_b.get(j, 0.0) for j in JOINTS], dtype=float)
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        # Cosine oscillation between A and B
        weights = 0.5 * (1.0 - np.cos(t))[:, None]
        loop = arr_a + weights * (arr_b - arr_a)
        return cls(name=name, beats=beats, loop=loop, ppb=ppb)


@dataclass
class ChoreographySegment:
    """A planned dance segment locked to bar downbeats."""
    start_beat: float
    move_name: str
    tempo: float
    label: str


class BaseRobotBus:
    """Abstract interface for robot motor communication."""
    def connect(self) -> bool:
        raise NotImplementedError
    def disconnect(self) -> None:
        raise NotImplementedError
    def write_pose(self, pose: Dict[str, float]) -> None:
        raise NotImplementedError
    def get_current_pose(self) -> Dict[str, float]:
        raise NotImplementedError


class LeRobotFeetechBus(BaseRobotBus):
    """Hardware driver using official LeRobot FeetechMotorsBus."""

    def __init__(self, port: str):
        self.port = port
        self.bus = None
        self.current_pose = {j: 0.0 for j in JOINTS}
        self.current_pose["gripper"] = 50.0

    def connect(self) -> bool:
        try:
            from lerobot.motors.feetech.feetech import FeetechMotorsBus, Motor, MotorNormMode
            motors = {
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            }
            self.bus = FeetechMotorsBus(port=self.port, motors=motors)
            self.bus.connect()
            return True
        except Exception as e:
            try:
                from lerobot.common.robot_devices.robots.feetech import FeetechMotorsBus
                self.bus = FeetechMotorsBus(port=self.port)
                self.bus.connect()
                return True
            except Exception as e2:
                print(f"[Warning] Could not initialize LeRobot FeetechMotorsBus on {self.port}: {e} / {e2}")
                return False

    def disconnect(self) -> None:
        if self.bus:
            try:
                self.bus.disconnect()
            except Exception:
                pass

    def write_pose(self, pose: Dict[str, float]) -> None:
        if not self.bus:
            return
        self.current_pose.update(pose)
        try:
            if hasattr(self.bus, "sync_write"):
                self.bus.sync_write("Goal_Position", pose)
            elif hasattr(self.bus, "write"):
                for joint, angle in pose.items():
                    self.bus.write("Goal_Position", {joint: angle})
        except Exception as err:
            print(f"[Warning] Motor write error: {err}")

    def get_current_pose(self) -> Dict[str, float]:
        return dict(self.current_pose)


class MockRobotBus(BaseRobotBus):
    """High-fidelity virtual bus for testing without hardware."""

    def __init__(self, port: str = "VIRTUAL"):
        self.port = port
        self.current_pose = {j: 0.0 for j in JOINTS}
        self.current_pose["gripper"] = 50.0
        self.connected = False

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False

    def write_pose(self, pose: Dict[str, float]) -> None:
        self.current_pose.update(pose)

    def get_current_pose(self) -> Dict[str, float]:
        return dict(self.current_pose)


class SO101Dancer:
    """High-clarity synchronized dance controller and choreography sequencer for SO-101."""

    # Default neutral and static anchor poses
    NEUTRAL_POSE = np.array([0.0, -90.0, 0.0, 60.0, 0.0, 0.0], dtype=float)

    def __init__(
        self,
        port: Optional[str] = None,
        force_sim: bool = False,
        speed_limit: float = 220.0,
        offset_ms: float = 70.0,
        moves_dir: str = "moves",
    ):
        self.force_sim = force_sim
        self.speed_limit = speed_limit
        self.offset_ms = offset_ms
        self.moves = self._load_moves(moves_dir)
        self.port, self.bus = self._init_bus(port, force_sim)
        self.visualizer = DanceVisualizer(
            mode="SIMULATION" if isinstance(self.bus, MockRobotBus) else "HARDWARE",
            port=str(self.port)
        )

    def _load_moves(self, moves_dir: str) -> Dict[str, MoveLoop]:
        """Loads dense 32-point-per-beat recorded trajectories, with synthetic backups."""
        loaded: Dict[str, MoveLoop] = {}
        paths = glob.glob(os.path.join(moves_dir, "*.json"))
        for p in paths:
            try:
                m = MoveLoop.load(p)
                loaded[m.name] = m
            except Exception as e:
                print(f"[Warning] Could not load move {p}: {e}")

        # Ensure core moves exist with high-quality synthetic backups if files missing
        if "sway" not in loaded:
            loaded["sway"] = MoveLoop.create_synthetic(
                "sway", 4,
                {"shoulder_pan": -45, "shoulder_lift": -95, "elbow_flex": 0, "wrist_flex": 60, "wrist_roll": -10, "gripper": 0},
                {"shoulder_pan": 45, "shoulder_lift": -95, "elbow_flex": 0, "wrist_flex": 60, "wrist_roll": 10, "gripper": 0}
            )
        if "bounce" not in loaded:
            loaded["bounce"] = MoveLoop.create_synthetic(
                "bounce", 4,
                {"shoulder_pan": -10, "shoulder_lift": -95, "elbow_flex": 5, "wrist_flex": 60, "wrist_roll": 0, "gripper": 0},
                {"shoulder_pan": -10, "shoulder_lift": -90, "elbow_flex": 35, "wrist_flex": 90, "wrist_roll": 0, "gripper": 0}
            )
        if "snap" not in loaded:
            loaded["snap"] = MoveLoop.create_synthetic(
                "snap", 4,
                {"shoulder_pan": -25, "shoulder_lift": -100, "elbow_flex": 50, "wrist_flex": 35, "wrist_roll": 15, "gripper": 0},
                {"shoulder_pan": 0, "shoulder_lift": -100, "elbow_flex": 60, "wrist_flex": 55, "wrist_roll": 65, "gripper": 50}
            )
        if "bigwave" not in loaded:
            loaded["bigwave"] = MoveLoop.create_synthetic(
                "bigwave", 8,
                {"shoulder_pan": -15, "shoulder_lift": -100, "elbow_flex": -70, "wrist_flex": -40, "wrist_roll": -5, "gripper": 0},
                {"shoulder_pan": -5, "shoulder_lift": 75, "elbow_flex": 35, "wrist_flex": 55, "wrist_roll": 25, "gripper": 80}
            )

        return loaded

    def _init_bus(self, port: Optional[str], force_sim: bool) -> Tuple[str, BaseRobotBus]:
        """Auto-detects hardware or selects simulation bus."""
        if force_sim:
            print("[INFO] Simulation mode selected. Using MockRobotBus.")
            return "VIRTUAL", MockRobotBus()

        detected_port = port or self.detect_serial_port()
        if detected_port:
            print(f"[INFO] Connecting to Feetech robot bus on {detected_port}...")
            hw_bus = LeRobotFeetechBus(detected_port)
            if hw_bus.connect():
                return detected_port, hw_bus
            print(f"[WARNING] Failed connecting to {detected_port}. Falling back to MockRobotBus.")

        print("[INFO] No robot hardware detected. Running in Simulation mode.")
        mock = MockRobotBus("VIRTUAL")
        mock.connect()
        return "VIRTUAL", mock

    @staticmethod
    def detect_serial_port() -> Optional[str]:
        """Scans for available serial ports on Windows, Linux, or macOS."""
        try:
            import serial.tools.list_ports
            ports = [p.device for p in serial.tools.list_ports.comports()]
            if ports:
                return ports[0]
        except Exception:
            pass

        for candidate in ["COM3", "COM4", "COM5", "/dev/ttyUSB0", "/dev/ttyACM0"]:
            if os.path.exists(candidate) if candidate.startswith("/") else False:
                return candidate
        return None

    def plan_choreography(self, events: List[BeatEvent], bpm: float) -> List[ChoreographySegment]:
        """Plans musical segments mapped to bars with adaptive energy-ranked phrasing for maximum clarity."""
        if not events:
            return [ChoreographySegment(0.0, "sway", 1.0, "SWAY")]

        bar_size = 4  # 4/4 meter
        total_beats = len(events)
        energies = np.array([e.energy for e in events], dtype=float)

        # Compute average energy per 4-beat bar
        bar_energies = [
            float(np.mean(energies[i:min(i + bar_size, total_beats)]))
            for i in range(0, total_beats, bar_size)
        ]

        if not bar_energies:
            return [ChoreographySegment(0.0, "sway", 1.0, "SWAY")]

        # Adaptive thresholding: songs with different gain/mastering span full dynamic range
        p35 = float(np.percentile(bar_energies, 35))
        p75 = float(np.percentile(bar_energies, 75))

        segments: List[ChoreographySegment] = []
        for i, avg_energy in enumerate(bar_energies):
            bar_events = events[i * bar_size:min((i + 1) * bar_size, total_beats)]
            has_drop = any(e.is_drop or e.section == "drop" for e in bar_events)

            # High-clarity movement selection based on adaptive music energy
            if has_drop or avg_energy >= p75:
                move_name = "bigwave" if "bigwave" in self.moves else "snap"
                label = "DROP WAVE"
            elif avg_energy >= p35:
                # Alternate between rhythmic bounce and snappy syncopation
                if ((i // 2) % 2 == 0) and "snap" in self.moves:
                    move_name = "snap"
                    label = "SNAP GROOVE"
                else:
                    move_name = "bounce" if "bounce" in self.moves else "sway"
                    label = "BOUNCE"
            else:
                move_name = "sway" if "sway" in self.moves else list(self.moves.keys())[0]
                label = "CHILL SWAY"

            start_beat = float(i * bar_size)

            # Condense consecutive identical moves so the arm preserves uninterrupted continuous loop flow
            if segments and segments[-1].move_name == move_name and segments[-1].label == label:
                continue

            segments.append(ChoreographySegment(
                start_beat=start_beat,
                move_name=move_name,
                tempo=1.0,
                label=label
            ))

        return segments

    def sample_pose(
        self,
        beat_pos: float,
        segments: List[ChoreographySegment],
        event: Optional[BeatEvent] = None
    ) -> Tuple[np.ndarray, str]:
        """Evaluates continuous trajectory with smoothstep crossfade between moves."""
        if not segments:
            return self.NEUTRAL_POSE.copy(), "NEUTRAL"

        # Find active segment
        seg_starts = [s.start_beat for s in segments]
        k = max(0, int(np.searchsorted(seg_starts, beat_pos, side="right")) - 1)
        curr_seg = segments[k]
        move = self.moves.get(curr_seg.move_name, list(self.moves.values())[0])

        phase = max(0.0, beat_pos - curr_seg.start_beat) * curr_seg.tempo
        pose = move.pose(phase).copy()

        # Crossfade from previous move over 1 beat
        if k > 0 and (beat_pos - curr_seg.start_beat) < FADE_BEATS:
            prev_seg = segments[k - 1]
            prev_move = self.moves.get(prev_seg.move_name, move)
            prev_phase = (beat_pos - prev_seg.start_beat) * prev_seg.tempo
            prev_pose = prev_move.pose(prev_phase)
            w = smoothstep((beat_pos - curr_seg.start_beat) / FADE_BEATS)
            pose = (1.0 - w) * prev_pose + w * pose

        # Micro-accent injection on beat transients (for razor-sharp rhythm)
        frac = beat_pos - np.floor(beat_pos)
        dist_to_beat = min(frac, 1.0 - frac)
        if dist_to_beat < 0.12 and event:
            energy_factor = min(1.0, max(0.2, event.energy))
            pulse = math.exp(-60.0 * (dist_to_beat ** 2)) * energy_factor
            if event.is_downbeat:
                # Downbeat emphasis: slight dip on kick
                pose[1] -= 3.0 * pulse  # shoulder_lift
                pose[2] += 4.0 * pulse  # elbow_flex
            elif int(np.round(beat_pos)) % 2 == 1:
                # Snare clap on beats 2 and 4
                pose[5] = min(100.0, pose[5] + 35.0 * pulse)  # gripper snap
                pose[3] += 5.0 * pulse  # wrist_flex nod

        return pose, curr_seg.label

    def glide(self, start_pose: np.ndarray, end_pose: np.ndarray, duration_sec: float) -> None:
        """Safely glides the arm from start_pose to end_pose to prevent mechanical shock."""
        n_steps = max(1, int(duration_sec * CONTROL_HZ))
        period = 1.0 / CONTROL_HZ
        for i in range(1, n_steps + 1):
            t0 = time.perf_counter()
            w = smoothstep(i / n_steps)
            q = (1.0 - w) * start_pose + w * end_pose
            self.bus.write_pose({j: float(v) for j, v in zip(JOINTS, q)})
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def play_dance(
        self,
        beat_events: List[Any],
        bpm: float,
        duration: float,
        audio_playback_fn: Optional[Any] = None,
        audio_stop_fn: Optional[Any] = None,
    ) -> None:
        """Executes the synchronized dance routine with continuous sub-beat tracking and live HUD."""
        # Convert simple timestamps to BeatEvent objects if necessary
        events: List[BeatEvent] = []
        for i, b in enumerate(beat_events):
            if isinstance(b, BeatEvent):
                events.append(b)
            else:
                events.append(BeatEvent(
                    time=float(b),
                    bpm=bpm,
                    energy=0.6,
                    is_downbeat=(i % 4 == 0),
                    is_drop=False,
                    section="groove"
                ))

        if not events:
            print("No beat events found to dance to.")
            return

        beat_times = np.array([e.time for e in events], dtype=float)
        segments = self.plan_choreography(events, bpm)
        print(f"High-Clarity Dance Sequencer initialized on [{self.port}].")
        print(f"Loaded Moves: {', '.join(self.moves.keys())} | Total Segments: {len(segments)}")

        # Continuous fractional beat position mapper
        def get_beat_pos(t: float) -> float:
            if t <= beat_times[0]:
                spb = max(beat_times[1] - beat_times[0], 0.1) if len(beat_times) > 1 else (60.0 / bpm)
                return (t - beat_times[0]) / spb
            if t >= beat_times[-1]:
                spb = max(beat_times[-1] - beat_times[-2], 0.1) if len(beat_times) > 1 else (60.0 / bpm)
                return (len(beat_times) - 1) + (t - beat_times[-1]) / spb
            return float(np.interp(t, beat_times, np.arange(len(beat_times))))

        # Read starting pose
        cur_dict = self.bus.get_current_pose()
        home_pose = np.array([cur_dict.get(j, self.NEUTRAL_POSE[i]) for i, j in enumerate(JOINTS)])
        offset_sec = self.offset_ms / 1000.0

        # Calculate first dance pose with anticipatory offset
        first_pose, first_label = self.sample_pose(get_beat_pos(offset_sec), segments, events[0])

        print(f"\n[INFO] Gliding in {GLIDE_S:.1f}s to initial dance pose...")
        self.glide(home_pose, first_pose, GLIDE_S)
        q_prev = first_pose.copy()

        # Start synchronized audio playback
        if audio_playback_fn:
            audio_playback_fn()

        start_time = time.time()
        max_step = self.speed_limit / CONTROL_HZ
        period = 1.0 / CONTROL_HZ

        try:
            from rich.live import Live
            with Live(console=self.visualizer.console, refresh_per_second=30) as live:
                while True:
                    tick_start = time.perf_counter()
                    elapsed = time.time() - start_time
                    if elapsed >= duration:
                        break

                    # Anticipatory timing: evaluate target ahead of acoustic transient
                    t_lookahead = elapsed + offset_sec
                    bp = get_beat_pos(t_lookahead)
                    cur_beat_idx = int(np.clip(np.floor(get_beat_pos(elapsed)), 0, len(events) - 1))
                    cur_event = events[cur_beat_idx]

                    # Sample continuous trajectory
                    want, move_label = self.sample_pose(bp, segments, cur_event)

                    # Hardware-safe velocity clipping
                    delta = want - q_prev
                    q = q_prev + np.clip(delta, -max_step, max_step)
                    q_prev = q

                    # Transmit to hardware or simulation
                    pose_dict = {j: float(v) for j, v in zip(JOINTS, q)}
                    self.bus.write_pose(pose_dict)

                    # Live Rich HUD
                    panel = self.visualizer.render_hud(
                        current_time=min(elapsed, duration),
                        duration=duration,
                        bpm=bpm,
                        beat_idx=cur_beat_idx,
                        total_beats=len(events),
                        pose_name=f"{move_label} (Phase: {int((bp % 4) / 4.0 * 100)}%)",
                        section=cur_event.section,
                        energy=cur_event.energy,
                        joints=pose_dict,
                    )
                    live.update(panel)

                    # Maintain 50 Hz control rate
                    dt = time.perf_counter() - tick_start
                    if dt < period:
                        time.sleep(period - dt)

        except KeyboardInterrupt:
            print("\n[INFO] Dance interrupted by user.")
        finally:
            if audio_stop_fn:
                try:
                    audio_stop_fn()
                except Exception:
                    pass
            print(f"\n[INFO] Gliding in {GLIDE_S:.1f}s back to safe home position...")
            self.glide(q_prev, home_pose, GLIDE_S)
            self.bus.disconnect()
            print("[INFO] Robot safely parked. Bus disconnected.")
