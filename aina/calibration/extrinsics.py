# Copyright (c) Meta Platforms, Inc. and affiliates.

import os

import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R

from aina.robot.camera_api import CameraAPI
from aina.robot.kinova_control import KinovaControl
from aina.utils.constants import (
    ARUCO_CENTER_TO_EEF,
    ARUCO_DICT,
    ARUCO_MARKER_ID,
    ARUCO_MARKER_SIZE,
    CALIBRATION_DATA_DIR,
)
from aina.utils.visualization import plot_points, project_poses

np.set_printoptions(precision=2, suppress=True)


class ExtrinsicsCalibration:
    def __init__(self, camera_names):
        self.camera_names = camera_names

        # Initialize the camera APIs
        self.camera_apis = {}
        for camera_name in self.camera_names:
            self.camera_apis[camera_name] = CameraAPI(camera_name)

        # Initialize the arm api
        print(f"Initializing arm api")
        self.arm_api = KinovaControl()
        print(f"Arm api initialized")

    def load_poses(self):
        arm_poses = np.load(f"{CALIBRATION_DATA_DIR}/arm_poses.npy")
        return arm_poses

    def get_images_and_arm_pose(self, visualize=False):
        """
        Method to get images of the environment from all the cameras
        """
        # Create the directory to store the images
        os.makedirs(CALIBRATION_DATA_DIR, exist_ok=True)

        arm_poses = []
        frame_id = 0
        while True:
            try:
                # timer.start_loop()
                input(f"Press Enter to continue to the image...")

                for camera_name in self.camera_names:
                    print(f"Getting image from {camera_name}")
                    rgb_image, _ = self.camera_apis[camera_name].get_rgb_image()
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(
                        f"{CALIBRATION_DATA_DIR}/{camera_name}_{frame_id:03d}.png",
                        rgb_image,
                    )

                arm_pose = self.arm_api.get_cartesian_pose()
                arm_poses.append(arm_pose)

                frame_id += 1

            except KeyboardInterrupt:
                break

        arm_poses = np.array(arm_poses)
        for camera_name in self.camera_names:
            self.camera_apis[camera_name].stop()

        print(
            f"Saving arm poses {arm_poses.shape} to {CALIBRATION_DATA_DIR}/arm_poses.npy"
        )
        np.save(f"{CALIBRATION_DATA_DIR}/arm_poses.npy", arm_poses)

    def get_aruco_corners_in_2d(self, frame_id, camera_name, img=None):

        # Load the image
        if img is None:
            img_path = f"{CALIBRATION_DATA_DIR}/{camera_name}_{frame_id:03d}.png"
            img = np.asarray(cv2.imread(img_path))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
        parameters = aruco.DetectorParameters()
        aruco_detector = aruco.ArucoDetector(
            dictionary=aruco_dict, detectorParams=parameters
        )

        corners, ids, _ = aruco_detector.detectMarkers(img_gray)

        hand_corners = None
        for i in range(len(corners)):
            if ids[i][0] == ARUCO_MARKER_ID:
                hand_corners = corners[i]
                break

        if hand_corners is None:
            return None
        return hand_corners.squeeze()

    def get_aruco_corners_in_3d(self, arm_pose):
        # position of three points in the end effector frame

        p1 = np.array(
            [
                ARUCO_CENTER_TO_EEF[0] - ARUCO_MARKER_SIZE / 2,
                ARUCO_CENTER_TO_EEF[1],
                ARUCO_CENTER_TO_EEF[2] + ARUCO_MARKER_SIZE / 2,
                1,
            ]
        )  # Upper left
        left_top_point = np.eye(4)
        left_top_point[:3, 3] = p1[:3]
        p2 = np.array(
            [
                ARUCO_CENTER_TO_EEF[0] + ARUCO_MARKER_SIZE / 2,
                ARUCO_CENTER_TO_EEF[1],
                ARUCO_CENTER_TO_EEF[2] + ARUCO_MARKER_SIZE / 2,
                1,
            ]
        )  # Upper right
        right_top_point = np.eye(4)
        right_top_point[:3, 3] = p2[:3]

        p3 = np.array(
            [
                ARUCO_CENTER_TO_EEF[0] + ARUCO_MARKER_SIZE / 2,
                ARUCO_CENTER_TO_EEF[1],
                ARUCO_CENTER_TO_EEF[2] - ARUCO_MARKER_SIZE / 2,
                1,
            ]
        )  # Lower right
        right_bottom_point = np.eye(4)
        right_bottom_point[:3, 3] = p3[:3]

        p4 = np.array(
            [
                ARUCO_CENTER_TO_EEF[0] - ARUCO_MARKER_SIZE / 2,
                ARUCO_CENTER_TO_EEF[1],
                ARUCO_CENTER_TO_EEF[2] - ARUCO_MARKER_SIZE / 2,
                1,
            ]
        )  # Lower left
        left_bottom_point = np.eye(4)
        left_bottom_point[:3, 3] = p4[:3]

        # transfer the cartesian position into rotation matrix
        eef_to_base = np.eye(4)
        eef_to_base[:3, :3] = R.from_euler(
            "xyz", arm_pose[3:], degrees=True
        ).as_matrix()
        eef_to_base[:3, 3] = arm_pose[:3]

        corner_1 = (eef_to_base @ left_top_point)[:3, 3]
        corner_2 = (eef_to_base @ right_top_point)[:3, 3]
        corner_3 = (eef_to_base @ right_bottom_point)[:3, 3]
        corner_4 = (eef_to_base @ left_bottom_point)[:3, 3]

        corners = np.stack((corner_1, corner_2, corner_3, corner_4))

        return corners

    def get_base_to_camera(self, camera_name):

        # Find the length of kinova_poses
        arm_poses = self.load_poses()
        print("in get_base_to_camera - arm_poses.shape: {}".format(arm_poses.shape))
        len_frames = arm_poses.shape[0]

        pts_3d = []
        pts_2d = []
        num_of_frames_missing = 0
        for frame_id in range(len_frames):
            curr_2d = self.get_aruco_corners_in_2d(frame_id, camera_name)  # (4,2) - A_C
            if curr_2d is None:
                num_of_frames_missing += 1
                continue
            arm_pose = arm_poses[frame_id]
            curr_3d = self.get_aruco_corners_in_3d(arm_pose)  # (4,3) - A_B

            pts_3d.append(curr_3d)
            pts_2d.append(curr_2d)

        pts_2d = np.concatenate(pts_2d, axis=0)
        pts_3d = np.concatenate(pts_3d, axis=0)

        # Solve this equation
        camera_intrinsics = self.camera_apis[camera_name].get_intrinsics()
        camera_distortion = self.camera_apis[camera_name].get_distortion_coefficients()
        retval, rvec, tvec = cv2.solvePnP(
            pts_3d,
            pts_2d,
            camera_intrinsics,
            camera_distortion,
            flags=cv2.SOLVEPNP_SQPNP,
        )

        rot_mtx = R.from_rotvec(rvec.squeeze()).as_matrix()
        homo_base_to_cam = np.eye(4)
        homo_base_to_cam[:3, :3] = rot_mtx
        homo_base_to_cam[:3, 3] = tvec.squeeze()
        cam_to_base = np.linalg.pinv(homo_base_to_cam)

        print(f"camera_name: {camera_name}")
        print("cam_to_base: {}".format(cam_to_base))
        print("missing frames!: {}".format(num_of_frames_missing))
        print("-------")

        np.save(f"{CALIBRATION_DATA_DIR}/{camera_name}_cam_to_base.npy", cam_to_base)

        return homo_base_to_cam  # H_B_C

    def get_aruco_corners_in_2d_from_robot(self, arm_pose, base_to_camera, camera_name):
        # First get the corners in 3d from the robot
        corners_to_base_tvecs = self.get_aruco_corners_in_3d(
            arm_pose=arm_pose
        )  # (4,3) H_A_B
        # Turn them into matrices just to project tha axes properly
        corners_to_base = np.stack(
            len(corners_to_base_tvecs) * [np.eye(4)], axis=0
        )  # (4,4,4)
        corners_to_base[:, :3, 3] = corners_to_base_tvecs[:, :3]

        # Take these corners to the camera
        corners_to_camera = base_to_camera @ corners_to_base

        # Project these corners
        projected_corners_in_camera = project_poses(
            poses=corners_to_camera,
            intrinsic_matrix=self.camera_apis[camera_name].get_intrinsics(),
            distortion=self.camera_apis[camera_name].get_distortion_coefficients(),
            scale=0,
        )[:, 3, :, :].squeeze()

        return projected_corners_in_camera

    def get_calibration_error_in_2d(self, camera_name, visualize=False):

        arm_poses = self.load_poses()
        print(arm_poses.shape)
        len_frames = arm_poses.shape[0]

        corner_errors = []

        base_to_camera = self.get_base_to_camera(camera_name)

        for frame_id in range(len_frames):

            corners_in_2d_from_camera = self.get_aruco_corners_in_2d(
                frame_id=frame_id, camera_name=camera_name
            )  # (4,2)
            if corners_in_2d_from_camera is None:
                continue

            corners_in_2d_from_robot = self.get_aruco_corners_in_2d_from_robot(
                arm_pose=arm_poses[frame_id],
                base_to_camera=base_to_camera,
                camera_name=camera_name,
            )  # (4,2)

            corner_diffs = np.linalg.norm(
                corners_in_2d_from_camera - corners_in_2d_from_robot, axis=-1
            )  # (4,)

            corner_errors.append(corner_diffs)

            if visualize:
                # Plot all the points
                img_path = f"{CALIBRATION_DATA_DIR}/{camera_name}_{frame_id:03d}.png"
                img = np.asarray(cv2.imread(img_path))
                img = plot_points(
                    corners_in_2d_from_camera,
                    img,
                    colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)],
                )
                img = plot_points(
                    corners_in_2d_from_robot,
                    img,
                    colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)],
                )
                cv2.imwrite(
                    f"{CALIBRATION_DATA_DIR}/{camera_name}_{frame_id:03d}_points.png",
                    img,
                )

                # visualizer.add_transform(transform=base_to_camera, time=frame_id, name_tag="kinova/base_to_camera")

        pixel_error_sep = np.mean(corner_errors, axis=0)  # (4,)

        pixel_error_all = np.mean(pixel_error_sep)  # (1,)

        print(
            "** CALIBRATION ERROR IN 2D - SEP CORNERS: {}, ALL CORNERS: {} **\n--------------------------------".format(
                pixel_error_sep, pixel_error_all
            )
        )
