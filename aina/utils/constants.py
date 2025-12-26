# Copyright (c) Meta Platforms, Inc. and affiliates.

import cv2.aruco as aruco
import numpy as np

from aina.utils.file_ops import get_repo_root

# Submodule paths
KINOVA_PATH = f"{get_repo_root()}/submodules/kinova-control/api_python/examples"
ABILITY_HAND_PATH = f"{get_repo_root()}/submodules/ability-control/python"
FOUNDATION_STEREO_PATH = f"{get_repo_root()}/submodules/FoundationStereo"
HAMER_PATH = f"{get_repo_root()}/submodules/hamer"

# Network ports
REALSENSE_CAMERA_IDS = {"left": "925622070557", "right": "934222072381"}
REALSENSE_TOPIC_PREFIXES = {
    "left": "/realsense/left_camera",
    "right": "/realsense/right_camera",
}
CAMERA_PORTS = {"left": 5555, "right": 5556, "gen3": 5557}
CAMERA_DEPTH_PORTS = {"left": 5558, "right": 5559, "gen3": 5560}
CAMERA_EXTRINSICS_PORTS = {"left": 5561, "right": 5562, "gen3": 5563}

# Calibration constants
ARUCO_MARKER_SIZE = 0.055
ARUCO_MARKER_ID = 0
ARUCO_DICT = aruco.DICT_4X4_50
ARUCO_CENTER_TO_EEF = np.array(
    [0, -0.0015, 0.1245]
)  # This is in the eef frame (the eef that you get from the arm)
CALIBRATION_DATA_DIR = f"{get_repo_root()}/aina/calibration/calibration_data"

# Intrinsics constants
LEFT_INTRINSICS = np.array(
    [
        [919.229, 0, 644.381],
        [0, 919.283, 357.638],
        [0, 0, 1],
    ]
)
RIGHT_INTRINSICS = np.array([[915.591, 0, 652.115], [0, 915.688, 369.995], [0, 0, 1]])


# Extrinsics
WRIST_TO_EEF = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.095], [0, 0, 0, 1]])
LEFT_TO_BASE = np.array(  # NOTE: This should be updated!!!
    [
        [-0.72, -0.56, 0.42, -0.01],
        [-0.69, 0.51, -0.51, 0.65],
        [0.07, -0.65, -0.75, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
RIGHT_TO_BASE = np.array(  # NOTE: This should be updated!!!
    [
        [0.96, 0.16, -0.22, 0.45],
        [0.27, -0.71, 0.65, -0.19],
        [-0.05, -0.69, -0.73, 0.51],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Robot constants
KINOVA_IP = "192.168.1.10"
ABILITY_MOTOR_LIMITS = [  # Joint limits in real life
    (0, 100),
    (0, 100),
    (0, 100),
    (0, 100),
    (0, 95),
    (0, -100),
]
