"""Robot controller and choreography engine for BeatSync SO-101.

Supports:
- Physical Feetech STS3215 motor bus via LeRobot or direct PySerial
- Automatic serial port discovery (Windows COM ports & Linux /dev/ttyUSB*)
- Simulated/Mock bus with real-time visualizer
- Dynamic choreographic pose selection based on musical dynamics
"""

import sys
import os
import time
import math
from typing import Dict, List, Optional, Tuple, Any

from src.audio_processor import BeatEvent
from src.visualizer import DanceVisualizer


# Standard joint names matching SO-101 / SO-ARM100
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


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
            # Fallback for earlier/custom LeRobot versions
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
    """Synchronized dance controller and choreography sequencer for the SO-101 robot arm."""

    # Rich choreography poses for 6-DOF expressive dance
    POSES: Dict[str, Dict[str, float]] = {
        "neutral": {
            "shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0,
            "wrist_flex": 0, "wrist_roll": 0, "gripper": 50
        },
        "beat_left": {
            "shoulder_pan": 35, "shoulder_lift": -12, "elbow_flex": 20,
            "wrist_flex": -20, "wrist_roll": 35, "gripper": 85
        },
        "beat_right": {
            "shoulder_pan": -35, "shoulder_lift": 12, "elbow_flex": -20,
            "wrist_flex": 20, "wrist_roll": -35, "gripper": 25
        },
        "drop_high": {
            "shoulder_pan": 0, "shoulder_lift": 40, "elbow_flex": 45,
            "wrist_flex": 25, "wrist_roll": 80, "gripper": 100
        },
        "bass_drop": {
            "shoulder_pan": 0, "shoulder_lift": -35, "elbow_flex": -35,
            "wrist_flex": -40, "wrist_roll": 0, "gripper": 100
        },
        "head_bang": {
            "shoulder_pan": 0, "shoulder_lift": -20, "elbow_flex": 35,
            "wrist_flex": 60, "wrist_roll": 0, "gripper": 70
        },
        "wave_left": {
            "shoulder_pan": 45, "shoulder_lift": 20, "elbow_flex": -15,
            "wrist_flex": 30, "wrist_roll": -50, "gripper": 40
        },
        "wave_right": {
            "shoulder_pan": -45, "shoulder_lift": -20, "elbow_flex": 15,
            "wrist_flex": -30, "wrist_roll": 50, "gripper": 40
        },
        "clap_snap": {
            "shoulder_pan": 0, "shoulder_lift": 10, "elbow_flex": 15,
            "wrist_flex": 0, "wrist_roll": 0, "gripper": 5
        },
    }

    def __init__(self, port: Optional[str] = None, force_sim: bool = False):
        self.force_sim = force_sim
        self.port, self.bus = self._init_bus(port, force_sim)
        self.visualizer = DanceVisualizer(
            mode="SIMULATION" if isinstance(self.bus, MockRobotBus) else "HARDWARE",
            port=str(self.port)
        )

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
        """Scans for available serial ports on Windows or Linux."""
        try:
            import serial.tools.list_ports
            ports = [p.device for p in serial.tools.list_ports.comports()]
            if ports:
                return ports[0]
        except Exception:
            pass

        # Check typical default ports
        for candidate in ["COM3", "COM4", "COM5", "/dev/ttyUSB0", "/dev/ttyACM0"]:
            if os.path.exists(candidate) if candidate.startswith("/") else False:
                return candidate
        return None

    def select_pose_for_event(self, event: BeatEvent, toggle: bool, beat_count: int) -> Tuple[str, Dict[str, float]]:
        """Intelligently chooses choreography based on musical energy and downbeats."""
        if event.is_drop or event.section == "drop":
            if beat_count % 4 == 0:
                pose_name = "drop_high"
            elif beat_count % 2 == 0:
                pose_name = "bass_drop"
            else:
                pose_name = "head_bang"
        elif event.section == "intro":
            # Subtle swaying
            pose_name = "wave_left" if toggle else "wave_right"
        elif event.energy > 0.5:
            # Active rhythm
            if beat_count % 4 == 2:
                pose_name = "clap_snap"
            else:
                pose_name = "beat_left" if toggle else "beat_right"
        else:
            # Chill groove
            pose_name = "beat_left" if toggle else "beat_right"

        return pose_name, self.POSES[pose_name]

    def play_dance(
        self,
        beat_events: List[Any],
        bpm: float,
        duration: float,
        audio_playback_fn: Optional[Any] = None
    ) -> None:
        """Executes the synchronized dance routine, updating motors and HUD."""
        print(f"Robot dancing initialized on [{self.port}]. Total beats: {len(beat_events)}")

        # Convert simple float timestamps to BeatEvent objects if needed
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
            print("No beat events to dance to.")
            return

        # Start optional synchronized audio playback
        if audio_playback_fn:
            audio_playback_fn()

        start_time = time.time()
        beat_idx = 0
        toggle = False
        current_pose_name = "neutral"
        current_pose_target = dict(self.POSES["neutral"])
        interpolated_pose = dict(self.POSES["neutral"])

        try:
            from rich.live import Live
            with Live(console=self.visualizer.console, refresh_per_second=30) as live:
                while beat_idx < len(events):
                    elapsed = time.time() - start_time
                    target_event = events[beat_idx]

                    # Trigger next pose on beat timestamp
                    if elapsed >= target_event.time:
                        current_pose_name, current_pose_target = self.select_pose_for_event(
                            target_event, toggle, beat_idx
                        )
                        toggle = not toggle
                        beat_idx += 1

                    # Smooth interpolation toward target pose (prevents jerky motor strain)
                    lerp_factor = 0.35  # Smooth responsiveness
                    for j in JOINTS:
                        cur_v = interpolated_pose[j]
                        target_v = current_pose_target.get(j, 0.0)
                        interpolated_pose[j] = cur_v + (target_v - cur_v) * lerp_factor

                    # Write to hardware or mock bus
                    self.bus.write_pose(interpolated_pose)

                    # Update live visualizer
                    current_event = events[min(beat_idx, len(events) - 1)]
                    panel = self.visualizer.render_hud(
                        current_time=min(elapsed, duration),
                        duration=duration,
                        bpm=bpm,
                        beat_idx=beat_idx,
                        total_beats=len(events),
                        pose_name=current_pose_name,
                        section=current_event.section,
                        energy=current_event.energy,
                        joints=interpolated_pose,
                    )
                    live.update(panel)

                    # 100 Hz control loop
                    time.sleep(0.01)

                # Return to neutral after routine
                for _ in range(30):
                    for j in JOINTS:
                        interpolated_pose[j] += (self.POSES["neutral"][j] - interpolated_pose[j]) * 0.2
                    self.bus.write_pose(interpolated_pose)
                    time.sleep(0.02)

        finally:
            self.bus.disconnect()
            print("\nDance routine finished safely. SO-101 in safe neutral position.")
