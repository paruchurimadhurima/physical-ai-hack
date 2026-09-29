# Developer & AI Agent Context Guide

> This document provides technical context, safety invariants, and architectural guidelines for **AI models, agents, and human developers** building on top of the BeatSync SO-101 project.

---

## 1. Quick Context for AI Models (LLMs & Subagents)

If you are an AI model tasked with modifying, refactoring, or extending this repository, observe the following rules:

### ⚠️ Non-Negotiable Invariants & Safety Constraints

1. **Always Use Trajectory Interpolation (LERP)**:
   - Physical Feetech STS3215 servos will strip plastic/metal gears or trigger over-current protection if commanded with instant step jumps.
   - Never send raw target angles directly to the hardware bus in a tight loop. Always use the exponential smoothing interpolator (`lerp_factor` in [robot_controller.py](file:///c:/Users/Madhurima%20%28Temp%29/Documents/projects/physical-ai-hack/src/robot_controller.py)).

2. **Mandatory Safe Parking (`finally:` Block)**:
   - The SO-101 arm is an un-counterbalanced 6-DOF manipulator. If motor torque is suddenly released while extended, the arm will collapse under gravity and hit the table.
   - Any dance execution loop **must** restore the arm to `neutral` pose before disconnecting or exiting.

3. **Strict Physical Joint Limits**:
   All joint values must adhere to the physical kinematic constraints:
   - `shoulder_pan`: `[-90.0, +90.0]` degrees
   - `shoulder_lift`: `[-60.0, +60.0]` degrees
   - `elbow_flex`: `[-90.0, +90.0]` degrees
   - `wrist_flex`: `[-90.0, +90.0]` degrees
   - `wrist_roll`: `[-90.0, +90.0]` degrees
   - `gripper`: `[0.0, 100.0]` percent (`0` = fully closed, `100` = fully open)

4. **Preserve Dual-Mode Compatibility (Hardware & Mock)**:
   - Code must run seamlessly without physical hardware attached. Any new bus feature or motor write method added to `LeRobotFeetechBus` must have an identical signature and behavior implemented in `MockRobotBus`.

---

## 2. Codebase Organization

```
physical-ai-hack/
├── .agents/
│   └── skills/
│       └── dance-controller/
│           └── SKILL.md            # Native Antigravity skill declaration for agent tool use
├── docs/
│   ├── ARCHITECTURE.md             # In-depth architectural designs and sequence diagrams
│   ├── DEVELOPER_GUIDE.md          # This developer and AI agent guide
│   └── HACK-INFO.md                # Hackathon rules, schedule, and judging criteria
├── src/
│   ├── __init__.py
│   ├── audio_processor.py          # Librosa audio feature extraction & demo synthesizer
│   ├── robot_controller.py         # Feetech motor bus, LERP engine, & pose state machine
│   └── visualizer.py               # Rich terminal HUD, joint telemetry, & ASCII arm model
├── tracks/                         # Audio files (WAV, MP3, etc.)
│   └── demo_beat.wav               # Built-in synthetic 120 BPM demo track
├── main.py                         # CLI entry point and background audio sync
├── requirements.txt                # Python package dependencies
├── README.md                       # Public-facing documentation
└── DEVELOPMENT.md                  # Quick pointer to this guide
```

---

## 3. How to Extend the Project

### Recipe A: Adding New Dance Poses
New choreographic poses are defined in `SO101Dancer.POSES` inside [src/robot_controller.py](file:///c:/Users/Madhurima%20%28Temp%29/Documents/projects/physical-ai-hack/src/robot_controller.py).

Each pose is a dictionary specifying target angles for all 6 joints:
```python
"disco_point": {
    "shoulder_pan": -45.0,   # degrees: negative = right, positive = left
    "shoulder_lift": 50.0,   # degrees: positive = upward lift
    "elbow_flex": 60.0,      # degrees: positive = upward extension
    "wrist_flex": 30.0,      # degrees: positive = wrist pitch up
    "wrist_roll": 70.0,      # degrees: wrist rotation
    "gripper": 10.0,         # percent: pinched gripper index finger
}
```

To incorporate the new pose into the choreography logic, update `select_pose_for_event()` in [src/robot_controller.py](file:///c:/Users/Madhurima%20%28Temp%29/Documents/projects/physical-ai-hack/src/robot_controller.py):
```python
if event.is_drop and beat_count % 8 == 0:
    return "disco_point", self.POSES["disco_point"]
```

---

### Recipe B: Live Microphone / Line-In Streaming
Currently, [AudioBeatProcessor](file:///c:/Users/Madhurima%20%28Temp%29/Documents/projects/physical-ai-hack/src/audio_processor.py) processes pre-recorded audio files. To support real-time audio input from a DJ booth or microphone:
1. Use `sounddevice` or `pyaudio` to capture a ring buffer of audio chunks (e.g. 512 samples at 44.1 kHz).
2. Compute instantaneous energy using RMS: $\text{RMS} = \sqrt{\frac{1}{N} \sum x[i]^2}$.
3. Run `librosa.onset.onset_detect` over a rolling 2-second window to emit `BeatEvent` objects to a thread-safe `queue.Queue`.
4. The `SO101Dancer.play_dance()` loop can pop from this queue instead of indexing a pre-computed array.

---

### Recipe C: ElevenLabs Voice Commentary / MC Mode
The hackathon provides ElevenLabs API credits. You can add an expressive AI "Hype Man" or MC:
1. In `src/audio_processor.py`, detect when a major drop is approaching (e.g. 4 beats ahead of `is_drop=True`).
2. Dispatch a non-blocking request to the ElevenLabs TTS API:
   ```python
   # e.g., "Get ready for the drop in 3... 2... 1!"
   ```
3. Play the generated voice clip through the secondary audio channel while the robot prepares with the `drop_high` buildup pose.

---

### Recipe D: Computer Vision & Interactive Mirroring
Connect a USB webcam to turn the dancer into an interactive robot:
1. Use `mediapipe.solutions.pose` to track human dancer wrist and elbow keypoints.
2. Map normalized human joint angles to SO-101 servo limits.
3. Blend human tracking angles with musical beat pulses using the existing LERP interpolator:
   $$\theta_{\text{final}} = (1 - w) \cdot \theta_{\text{human}} + w \cdot \theta_{\text{beat}}$$

---

## 4. Hardware Connection & Troubleshooting

### Hardware Setup
- **Robot Arm**: Hugging Face / Feetech SO-101 (SO-ARM100) follower arm.
- **Servos**: 6x Feetech STS3215 serial bus smart servos (daisy-chained).
- **Interface**: Waveshare Bus Linker, Feetech U2D2, or ESP32 serial bridge connected via USB.
- **Power**: External DC power supply (7.4V to 12V, minimum 4A). **Do not power servos from USB alone.**

### Troubleshooting Serial Ports
- **Windows**: Check Device Manager under "Ports (COM & LPT)" or run:
  ```powershell
  [System.IO.Ports.SerialPort]::GetPortNames()
  ```
- **Linux**: Check `/dev/ttyUSB0` or `/dev/ttyACM0`. Ensure your user has dialout permissions:
  ```bash
  sudo usermod -a -G dialout $USER
  ```

---

## 5. Automated Validation & Test Suite

Verify changes without physical hardware using the simulation test pipeline:

```bash
# 1. Test audio feature extraction & synthetic beat generator
python -c "from src.audio_processor import AudioBeatProcessor; p = AudioBeatProcessor.generate_demo_track('tracks/test_run.wav', duration_sec=4.0); b = AudioBeatProcessor('tracks/test_run.wav'); bpm, ev = b.extract_expressive_beats(); assert len(ev) > 0; print('Audio tests passed! BPM:', bpm)"

# 2. Test headless / simulation dance execution (runs terminal HUD)
python main.py --demo --sim --no-sound
```
