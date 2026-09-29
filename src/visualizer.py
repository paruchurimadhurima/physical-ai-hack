"""Terminal visualizer for BeatSync SO-101 using rich."""

import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Ensure UTF-8 support on Windows terminals
if sys.platform == "win32":
    try:
        if sys.stdout.encoding.lower() != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr.encoding.lower() != "utf-8":
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class DanceVisualizer:
    """Renders a real-time HUD and 6-DOF joint telemetry for SO-101."""

    def __init__(self, mode: str = "SIMULATION", port: str = "VIRTUAL"):
        self.console = Console(force_terminal=True, highlight=False)
        self.mode = mode
        self.port = port

    def render_hud(
        self,
        current_time: float,
        duration: float,
        bpm: float,
        beat_idx: int,
        total_beats: int,
        pose_name: str,
        section: str,
        energy: float,
        joints: dict,
    ) -> Panel:
        """Constructs a rich panel showing current robot dance telemetry."""
        
        # Header Info
        header_text = Text()
        header_text.append(" [BEATSYNC SO-101] ", style="bold magenta reverse")
        header_text.append(f"  Mode: [{self.mode}] ", style="bold green" if self.mode == "HARDWARE" else "bold yellow")
        header_text.append(f" Port: {self.port}\n", style="cyan")

        # Track & Beat Metrics
        metrics_table = Table.grid(padding=(0, 2))
        metrics_table.add_column("Key", style="bold white")
        metrics_table.add_column("Value", style="bold cyan")
        metrics_table.add_column("Key2", style="bold white")
        metrics_table.add_column("Value2", style="bold yellow")

        time_str = f"{current_time:05.2f}s / {duration:05.2f}s"
        beat_str = f"{beat_idx + 1}/{total_beats}"
        energy_bar = "#" * int(energy * 15) + "-" * (15 - int(energy * 15))

        section_style = "bold red" if section == "drop" else ("bold green" if section == "groove" else "bold blue")

        metrics_table.add_row("Time:", time_str, "Energy:", f"[{energy_bar}] {int(energy * 100)}%")
        metrics_table.add_row("BPM:", f"{bpm:.1f}", "Pose:", f"{pose_name}")
        metrics_table.add_row("Beat:", beat_str, "Section:", Text(section.upper(), style=section_style))

        # Joint Angles Display
        joints_table = Table(title="6-DOF Motor Bus Telemetry", expand=True, show_edge=False, box=None)
        joints_table.add_column("Joint", style="bold white", width=14)
        joints_table.add_column("Angle / State", justify="center", width=14)
        joints_table.add_column("Position Gauge", ratio=1)

        joint_ranges = {
            "shoulder_pan": (-90, 90, "deg"),
            "shoulder_lift": (-110, 90, "deg"),
            "elbow_flex": (-90, 90, "deg"),
            "wrist_flex": (-90, 110, "deg"),
            "wrist_roll": (-90, 90, "deg"),
            "gripper": (0, 100, "%"),
        }

        for j_name, (min_v, max_v, unit) in joint_ranges.items():
            val = float(joints.get(j_name, 0))
            clamped = max(min_v, min(max_v, val))
            pct = (clamped - min_v) / (max_v - min_v) if max_v > min_v else 0.5
            
            bar_len = 24
            active_chars = int(pct * bar_len)
            gauge = "=" * active_chars + "O" + "-" * max(0, bar_len - active_chars - 1)
            
            color = "cyan" if "shoulder" in j_name else ("yellow" if "elbow" in j_name else "magenta")
            joints_table.add_row(
                j_name,
                f"{val:+5.1f} {unit}" if unit == "deg" else f"{val:4.1f} {unit}",
                Text(f"[{min_v:3.0f}] {gauge} [{max_v:3.0f}]", style=color)
            )

        # ASCII Miniature Arm Kinematics Visualization
        arm_ascii = self._generate_arm_ascii(joints)

        # Main Panel Content
        content = Table.grid(padding=1)
        content.add_row(header_text)
        content.add_row(metrics_table)
        content.add_row(joints_table)
        content.add_row(arm_ascii)

        return Panel(content, border_style="magenta" if section == "drop" else "blue", title="[bold]SO-101 Dancer HUD[/bold]")

    def _generate_arm_ascii(self, joints: dict) -> Text:
        """Renders an expressive ASCII posture representation of the arm."""
        pan = joints.get("shoulder_pan", 0)
        lift = joints.get("shoulder_lift", 0)
        elbow = joints.get("elbow_flex", 0)
        grip = joints.get("gripper", 50)

        grip_char = "< >" if grip > 60 else "<|>"
        sway = "   \\" if pan > 15 else (" /" if pan < -15 else "  |")
        lift_icon = " ^" if lift > 15 else (" v" if lift < -15 else " -")

        lines = [
            f"       [End-Effector]  {grip_char} (Gripper: {grip:.0f}%)",
            f"              \\     /   (Wrist Pitch/Roll: {joints.get('wrist_flex', 0):+.0f} deg / {joints.get('wrist_roll', 0):+.0f} deg)",
            f"             (Elbow: {elbow:+.0f} deg)",
            f"            {sway}  {lift_icon} (Shoulder Lift: {lift:+.0f} deg)",
            f"           [Base] (Pan: {pan:+.0f} deg)",
            f"       ==============="
        ]
        return Text("\n".join(lines), style="bold green")
