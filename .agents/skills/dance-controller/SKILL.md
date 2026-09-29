---
name: dance-controller
description: Executes audio-synchronized dance routines on the SO-101 robot arm. Use when the user asks to analyze beats, run a dance, test robot motor positions, or simulate choreography.
---

# SO-101 Dance Control Skill

This skill controls the Hugging Face / Feetech SO-101 (SO-ARM100) 6-DOF follower arm to dance in real-time synchronization with audio tracks.

## Overview
- **Audio Processing**: Uses `librosa` to analyze audio files, computing BPM, onset envelopes, beat drop timestamps, and frequency energy bands (bass vs mid/treble).
- **Choreography Engine**: Dynamically maps beat timing and musical intensity to 6-DOF joint angle keyframes (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`).
- **Hardware Bus & Simulation**: Direct control over Feetech STS3215 servos via `FeetechMotorsBus` with automatic serial port detection (`COMx` on Windows, `/dev/ttyUSB*` on Linux). Includes rich mock simulation if hardware is disconnected.

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
If no physical arm is plugged in, the controller automatically defaults to `--sim` (simulation) mode with a live terminal animation.

### 2. Audio Beat Extraction & Dance Execution
Run full audio analysis and dance routine:
```bash
python main.py --audio tracks/song.mp3 --port COM3
```

To run in Simulation / Visualizer mode (no hardware required):
```bash
python main.py --audio tracks/song.mp3 --sim
```

To generate a sample beat track and test immediately:
```bash
python main.py --demo --sim
```

### 3. Poses & Joint Mapping
The SO-101 uses 6 joints:
| Joint Name | Range (deg) | Role in Dance |
| :--- | :--- | :--- |
| `shoulder_pan` | -90° to +90° | Base sway, left/right swing |
| `shoulder_lift`| -60° to +60° | Vertical bobbing, body rise/fall |
| `elbow_flex`   | -90° to +90° | Forearm rhythm, bounce accent |
| `wrist_flex`   | -90° to +90° | Head/wrist nods, tempo accent |
| `wrist_roll`   | -90° to +90° | Flairs, waving, dramatic drops |
| `gripper`      | 0 (closed) to 100 (open) | Rhythmic claps, beat snaps |

---

## Troubleshooting
- **No serial port found**: Check USB connection on the U2D2 / Bus Linker adapter. Pass `--sim` to test choreography in software.
- **Permission denied on port (Linux)**: Run `sudo usermod -a -G dialout $USER` and replug USB.
- **Audio sync drift**: Use `--bpm` override if the track has a fixed known tempo, or check onset energy threshold in `src/audio_processor.py`.
