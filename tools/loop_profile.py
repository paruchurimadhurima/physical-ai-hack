"""Times every stage of the lerobot-rollout control loop WITHOUT moving the arm.

Rebuilds the same per-tick pipeline as `lerobot-rollout --strategy.type=base`
(read motors -> read camera -> build frame -> preprocess -> policy -> postprocess)
but never sends a goal position to the motors. Use it to find out which stage
pushes the loop below 30 Hz.

Usage:
    python tools/loop_profile.py [policy_path] [device] [--no-camera]
    # defaults: the cloud test model, device "mps"; try "cpu" for comparison.
    # --no-camera feeds a synthetic image instead (for shells without camera access).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

NO_CAMERA = "--no-camera" in sys.argv
args = [a for a in sys.argv[1:] if a != "--no-camera"]
POLICY = args[0] if len(args) > 0 else "chrislaubenthal/act_cloudtest_2026-09-28_20-23-26"
DEVICE = args[1] if len(args) > 1 else "mps"
PORT = "/dev/tty.usbmodem5B8E1123681"
CALIB = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/follower.json"
DATASET = "chrislaubenthal/so101_white_can_v3"
TASK = "Grip the white can and put it in the bowl"
TICKS, FPS = 300, 30

names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
calib = {k: MotorCalibration(**v) for k, v in json.loads(CALIB.read_text()).items()}
bus = FeetechMotorsBus(
    port=PORT,
    motors={
        n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if n == "gripper" else MotorNormMode.DEGREES)
        for i, n in enumerate(names)
    },
    calibration=calib,
)
cam = None if NO_CAMERA else OpenCVCamera(OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30))
fake_img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)

meta = LeRobotDatasetMetadata(DATASET)
cfg = PreTrainedConfig.from_pretrained(POLICY)
cfg.pretrained_path = POLICY
cfg.device = DEVICE
policy = make_policy(cfg, ds_meta=meta)
policy.eval()
pre, post = make_pre_post_processors(
    cfg, pretrained_path=POLICY, preprocessor_overrides={"device_processor": {"device": DEVICE}}
)
device = torch.device(DEVICE)


def sync():
    if DEVICE == "mps":
        torch.mps.synchronize()


stages = {k: [] for k in ["motoren", "kamera", "frame", "vorverarb.", "policy", "nachverarb.", "gesamt"]}
bus.connect()
if cam:
    cam.connect()
try:
    time.sleep(1.0)
    policy.reset()
    for tick in range(TICKS):
        t0 = time.perf_counter()
        state = bus.sync_read("Present_Position")
        t1 = time.perf_counter()
        img = cam.read_latest() if cam else fake_img.copy()
        t2 = time.perf_counter()
        frame = {
            "observation.state": np.array([state[n] for n in names], dtype=np.float32),
            "observation.images.front": img,
        }
        with torch.inference_mode():
            obs = prepare_observation_for_inference(frame, device, TASK, "so101_follower")
            sync()
            t3 = time.perf_counter()
            obs = pre(obs)
            sync()
            t4 = time.perf_counter()
            action = policy.select_action(obs)
            sync()
            t5 = time.perf_counter()
            action = post(action).squeeze(0).cpu()
        t6 = time.perf_counter()
        for k, dt in zip(stages, [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t6 - t0]):
            stages[k].append(dt * 1000)
        if (rest := 1 / FPS - (t6 - t0)) > 0:
            time.sleep(rest)
finally:
    if cam:
        cam.disconnect()
    bus.disconnect(disable_torque=False)

tot = np.array(stages["gesamt"])
print(f"\nPolicy: {POLICY} | Geraet: {DEVICE} | Kamera: {'Testbild' if NO_CAMERA else 'echt'} | {TICKS} Durchlaeufe (Ziel: < 33 ms)")
for k, v in stages.items():
    v = np.array(v)
    print(f"  {k:12s} median {np.median(v):6.1f} ms | p95 {np.percentile(v, 95):6.1f} ms | max {v.max():6.1f} ms")
print(f"  -> Durchlaeufe ueber 33 ms: {np.mean(tot > 1000 / FPS) * 100:.0f} %")
