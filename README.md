# BeatSync SO-101: Expressive Audio-Driven Robotic Dancer 🕺🤖

BeatSync SO-101 turns the **Hugging Face / Feetech SO-101 Follower Arm (6-DOF)** into an expressive robotic dancer that analyzes music in real-time and choreographs dynamic physical movements synchronized to beats, drops, and musical energy.

Built for the **Physical AI Hackathon (DARE Campus Zurich)**.

---

## ✨ Features

- **Expressive 6-DOF Choreography**: Maps musical energy, drops, and downbeats to full arm kinematics (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, and rhythmic `gripper` claps).
- **Audio Dynamic Extraction (`librosa`)**: Real-time onset strength, BPM tempo tracking, spectral centroid (bass vs treble), and section classification (`intro`, `groove`, `drop`, `chill`).
- **Smooth Trajectory Interpolation**: Eliminates jerky servo wear by smoothing joint transitions between beat points.
- **Physical + Simulation Modes**: Auto-detects serial COM ports (`COMx` on Windows, `/dev/ttyUSB*` on Linux). If hardware is not connected, smoothly falls back to simulation mode with an ASCII/terminal HUD.
- **Antigravity Custom Skill**: Registered under `.agents/skills/dance-controller/SKILL.md` for native Antigravity agent control.
- **Built-in Electronic Demo Synthesizer**: Generates a punchy synth kick/snare/drop track out of the box (`--demo`).

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Instant Demo (Simulation Mode)
No hardware or MP3 required — synthesizes a 120 BPM electronic track and renders live telemetry:
```bash
python main.py --demo --sim
```

### 3. Run Physical Robot Dance
Connect your SO-101 via USB Bus Linker / U2D2 and provide your favorite music track:
```bash
# Auto-detects COM port:
python main.py --audio tracks/song.mp3

# Or specify your port explicitly:
python main.py --audio tracks/song.mp3 --port COM3
```

---

## 🕹️ CLI Options

| Flag | Description |
| :--- | :--- |
| `--audio <path>` | Path to audio file (MP3, WAV, FLAC, etc.) |
| `--demo` | Synthesizes a kick/snare beat track with drops for instant testing |
| `--port <port>` | Serial port for Feetech bus (`COM3`, `/dev/ttyUSB0`) |
| `--sim` | Run in simulation mode with live HUD (no robot needed) |
| `--no-sound` | Mute speaker audio playback during dance |
| `--bpm <float>` | Manual BPM tempo override |

---

## 🤖 SO-101 Joint Mapping

| Joint Name | Servo ID | Normal Range | Role in Choreography |
| :--- | :---: | :---: | :--- |
| `shoulder_pan` | 1 | -90° to +90° | Base sway, left/right swing |
| `shoulder_lift`| 2 | -60° to +60° | Vertical bobbing, body rise/fall |
| `elbow_flex`   | 3 | -90° to +90° | Forearm rhythm, bounce accent |
| `wrist_flex`   | 4 | -90° to +90° | Head nods, tempo pulse |
| `wrist_roll`   | 5 | -90° to +90° | Wrist flair, dramatic drop sweeps |
| `gripper`      | 6 | 0 to 100% | Rhythmic claps and snaps |

---

## 🏗️ How It's Built & Technical Design

BeatSync SO-101 connects high-level music feature analysis with low-level smart servo kinematics:
1. **Audio Feature Extraction (`src/audio_processor.py`)**: Uses `librosa` to compute normalized onset strength envelopes, tempo tracking ($BPM$), and spectral centroids (sub-bass kick vs treble hi-hat separation) to generate a timestamped sequence of `BeatEvent` objects with drop detection.
2. **Kinematic Choreography Engine (`src/robot_controller.py`)**: A dynamic state machine selects expressive 6-DOF poses (`POSES`) conditioned on musical intensity, downbeats, and drops.
3. **Smooth Trajectory LERP**: An exponential linear interpolation filter running at 100 Hz prevents mechanical stress and guarantees fluid, lifelike robotic dancing.
4. **Dual Bus Support**: Seamlessly transitions between physical Feetech STS3215 servos (`LeRobotFeetechBus`) and an in-memory virtual bus (`MockRobotBus`).

For deep architectural documentation, sequence diagrams, and model guides:
- 📖 [System Architecture & Sequence Diagrams](docs/ARCHITECTURE.md)
- 🤖 [Developer & AI Agent Context Guide](docs/DEVELOPER_GUIDE.md) (Extension recipes, safety rules, and pose definitions)
- 🏆 [Hackathon Rules & Schedule](docs/HACK-INFO.md)

---

## 📂 Repository Structure

```
├── .agents/
│   └── skills/
│       └── dance-controller/
│           └── SKILL.md         # Native Antigravity Skill Definition
├── docs/
│   ├── ARCHITECTURE.md          # Technical architecture & sequence diagrams
│   ├── DEVELOPER_GUIDE.md       # AI Model & Developer context and recipes
│   └── HACK-INFO.md             # Hackathon rules, schedule & judging
├── src/
│   ├── __init__.py
│   ├── audio_processor.py       # Librosa beat, onset, and section analyzer
│   ├── robot_controller.py      # Feetech motor bus & choreography engine
│   └── visualizer.py            # Rich terminal telemetry and HUD
├── tracks/                      # Audio tracks folder
├── DEVELOPMENT.md               # Quick development entrypoint
├── main.py                      # Execution entrypoint
├── requirements.txt             # Python dependencies
└── README.md
```

