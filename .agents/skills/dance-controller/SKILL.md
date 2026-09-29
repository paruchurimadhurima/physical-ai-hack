---
name: dance-controller
description: Executes audio-synchronized dance routines on the SO-101 robot arm. Use when the user asks to analyze beats, run a dance, test robot motor positions, or simulate choreography.
---

# SO-101 Dance Control Skill

This skill controls the Hugging Face / Feetech SO-101 (SO-ARM100) 6-DOF follower arm to dance in real-time synchronization with audio tracks.

## Overview
- **Audio Processing**: Uses `librosa` to analyze audio files, computing BPM, onset envelopes, beat drop timestamps, and frequency energy bands (bass vs mid/treble).
- **Choreography Engine**: Continuous 32-point-per-beat trajectory playback using recorded motion loops (`moves/sway.json`, `moves/bounce.json`, `moves/snap.json`, `moves/bigwave.json`) with 1-beat smoothstep crossfading and anticipatory latency compensation (`--offset-ms`).
- **Hardware Bus & Simulation**: Direct control over Feetech STS3215 servos via `FeetechMotorsBus` with automatic serial port detection (`COMx` on Windows, `/dev/ttyUSB*` on Linux). Includes rich 30 FPS mock simulation if hardware is disconnected.
- **Hardware Diagnostics**: Read-only servo ping and voltage/temperature checks via `tools/servo_info.py`.

---

## Execution Runbook

### 1. Verify Motor & Serial Port Connection
On Windows, check available COM ports:
```powershell
powershell -Command "[System.IO.Ports.SerialPort]::GetPortNames()"
```
On Linux / macOS:
```bash
ls /dev/ttyUSB* /dev/ttyACM*
```
Run hardware bus diagnostics to verify servo communication, voltages, and temps:
```bash
python tools/servo_info.py COM3
```

### 2. Audio Beat Extraction & High-Clarity Dance Execution
Run full audio analysis and dance routine on hardware:
```bash
python main.py --audio inputs/pdoom.wav --port COM3 --offset-ms 70
```

To run in Simulation / Visualizer mode (no hardware required):
```bash
python main.py --audio inputs/pdoom.wav --sim
```

To generate a sample beat track and test immediately:
```bash
python main.py --demo --sim
```

### 3. Move Loops & Joint Envelope
The controller executes dense 32-point-per-beat recorded trajectories:
| Move Loop | Beats | Style / Musical Role | Primary Joints |
| :--- | :---: | :--- | :--- |
| `sway` | 4 | Chill intro/verse, smooth hip sway | `shoulder_pan` (-61° to +36°) |
| `bounce` | 4 | Groove rhythm, upbeat bounce | `elbow_flex` & `wrist_flex` |
| `snap` | 4 | High energy, syncopated rolls & claps | `wrist_roll` & `gripper` (0–47%) |
| `bigwave` | 8 | Climactic drops, dramatic full-body wave | Full 6-DOF vertical & pitch sweeps |

---

## Troubleshooting
- **No serial port found**: Check USB connection on the U2D2 / Bus Linker adapter. Pass `--sim` to test choreography in software.
- **Motion lags behind drum transients**: Increase `--offset-ms` (e.g. `--offset-ms 90`) to compensate for servo inertia.
- **Permission denied on port (Linux)**: Run `sudo usermod -a -G dialout $USER` and replug USB.

