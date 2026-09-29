"""Measures the real frame rate of the camera exactly as lerobot opens it.

Prints fps and image brightness once per second for 10 seconds. Try switching a
lamp on and off while it runs: many USB webcams halve their frame rate in dim
light (longer auto-exposure), which slows the whole lerobot control loop.

Usage:
    python tools/camera_fps.py [index]   # default index 0
"""

import sys
import time

import numpy as np
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=index, width=640, height=480, fps=30))
cam.connect()
try:
    for second in range(1, 11):
        frames, brightness = 0, []
        t_end = time.perf_counter() + 1.0
        while time.perf_counter() < t_end:
            brightness.append(cam.read().mean())
            frames += 1
        status = "OK" if frames >= 27 else "ZU LANGSAM"
        print(f"Sekunde {second:2d}: {frames:2d} fps | Helligkeit {np.mean(brightness):5.1f} / 255 | {status}")
finally:
    cam.disconnect()
