# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch

from aina.preprocessing.aria.vrs_demo import VRSDemo
from aina.preprocessing.point_utils.foundation_stereo_wrapper import (
    FoundationStereoWrapper,
)
from aina.utils.file_ops import suppress
from aina.utils.points import depth_to_xyz_map, transform_points_batch
from aina.utils.vision import rectify_image_with_kb


class VRSDepthExtractor:
    def __init__(self, vrs_demo: VRSDemo):
        self.vrs_demo = vrs_demo

        with suppress(stdout=True):
            self.foundation_stereo_wrapper = FoundationStereoWrapper()

        # Initialize the rectified camera matrices
        camera_extrinsics = self.vrs_demo.get_camera_projections()["extrinsics"]
        left_to_device = camera_extrinsics["slam-front-left"]
        right_to_device = camera_extrinsics["slam-front-right"]
        H_R_L = np.linalg.pinv(left_to_device) @ right_to_device
        self.R_L_NL, self.R_R_NR = self._create_scanline_rectified_cameras(H_R_L)

    # TODO: Move these to vision_utils.py
    def _create_scanline_rectified_cameras(self, H_R_L):
        """
        Returns the new rotation and translation matrices between the original and
        rectified camera frames for both left and right cameras, given the right-to-left homogenous matrix.
        """

        # Split the right-to-left homogenous matrix
        R_R_L = H_R_L[:3, :3]  # Rotation from right to left
        T_R_L = H_R_L[:3, 3]  # Translation from right to left

        # Up vector in left FoR
        lup_l = np.array([0.0, -1.0, 0.0])

        # Hypothetical forward vector, perpendicular to baseline
        r_l_norm = T_R_L / np.linalg.norm(T_R_L)  # Normalized baseline (baseline)
        lfwd_l = np.cross(
            lup_l, r_l_norm
        )  # Hypothetical forward vector, perpendicular to baseline
        if np.linalg.norm(lfwd_l) < 1e-6:  # singular case
            lfwd_l = np.array([0.0, 0.0, 1.0])
        lfwd_l = lfwd_l / np.linalg.norm(
            lfwd_l
        )  # Normalized hypothetical forward vector

        avgfwd_l = lfwd_l

        # Define new basis (in left FoR)
        nx_l = r_l_norm
        nz_l = avgfwd_l
        ny_l = np.cross(nz_l, nx_l)
        ny_l /= np.linalg.norm(ny_l)

        # New orientation for both left and right cameras (expressed relative to original left)
        R_L_NL = np.column_stack((nx_l, ny_l, nz_l))  # Rotation from left to new left
        R_R_NR = np.linalg.pinv(R_R_L) @ R_L_NL  # Rotation from right to new right

        return R_L_NL, R_R_NR

    def _rectify_image(self, image, camera_projection_params, R_old_to_new):

        # Create the new intrinsics matrix
        focal_scale = 1.25
        K_new = np.array(
            [
                [
                    camera_projection_params[0] * focal_scale,
                    0,
                    camera_projection_params[1],
                ],
                [
                    0,
                    camera_projection_params[0] * focal_scale,
                    camera_projection_params[2],
                ],
                [0, 0, 1],
            ]
        )

        # Get the direction ray in the original camera frame
        Rn_Kinv = torch.from_numpy(R_old_to_new @ np.linalg.inv(K_new)).to(
            "cuda", dtype=torch.float32
        )
        image = torch.from_numpy(image).to("cuda", dtype=torch.float32)
        params = torch.tensor(
            camera_projection_params, device=image.device, dtype=torch.float32
        )
        new_img = rectify_image_with_kb(image, Rn_Kinv, params, num_k=6)

        new_img = new_img.cpu().numpy()

        return new_img, K_new

    def get_rectified_slam_frames(self, frame_id: int):
        """
        Method to get the rectified slam frames for the given frame id.
        """

        # Get the slam frames
        slam_frames = self.vrs_demo.get_slam_frames(frame_id)

        # Projection params
        projection_params = self.vrs_demo.get_camera_projections()["projection_params"]

        # Rectify the left and right images
        rectified_left, K_new_left = self._rectify_image(
            image=slam_frames["left"],
            camera_projection_params=projection_params["slam-front-left"],
            R_old_to_new=self.R_L_NL,
        )
        rectified_right, K_new_right = self._rectify_image(
            image=slam_frames["right"],
            camera_projection_params=projection_params["slam-front-right"],
            R_old_to_new=self.R_R_NR,
        )

        # Update the camera intrinsics
        self.vrs_demo.update_camera_intrinsics(
            camera_label="slam-front-left", new_intrinsics=K_new_left
        )
        self.vrs_demo.update_camera_intrinsics(
            camera_label="slam-front-right", new_intrinsics=K_new_right
        )

        return dict(left=rectified_left.squeeze(), right=rectified_right.squeeze())

    def get_xyz_map_in_new_left(self, frame_id: int):
        """
        Method to get the xyz map for the given frame id.
        """

        # Get the rectified slam frames
        rectified_slam_frames = self.get_rectified_slam_frames(frame_id)

        # Get the disparity
        with suppress(stdout=True):
            disparity = self.foundation_stereo_wrapper.get_disparity(
                rectified_slam_frames["left"], rectified_slam_frames["right"]
            )

        # Get the baseline and new left intrinsics
        new_left_intrinsics = self.vrs_demo.get_camera_projections()["intrinsics"][
            "slam-front-left"
        ]
        baseline = self.vrs_demo.get_slam_camera_baseline()

        # Get the xyz map
        depth = new_left_intrinsics[0, 0] * baseline / disparity
        xyz_map = depth_to_xyz_map(depth, new_left_intrinsics, zmin=0.005)

        return xyz_map

    def get_xyz_map_in_rgb(self, frame_id: int):
        """
        Method to get the xyz map in the rgb frame for the given frame id.
        """
        # Get the xyz map in the new left frame
        xyz_map_in_new_left = self.get_xyz_map_in_new_left(frame_id)
        rgb_to_device = self.vrs_demo.get_camera_projections()["extrinsics"][
            "camera-rgb"
        ]
        left_to_device = self.vrs_demo.get_camera_projections()["extrinsics"][
            "slam-front-left"
        ]
        H_L_RGB = np.linalg.pinv(rgb_to_device) @ left_to_device
        H_L_NL = np.eye(4)
        H_L_NL[:3, :3] = self.R_L_NL
        H_NL_RGB = H_L_RGB @ H_L_NL

        # Transform the xyz map to the rgb frame
        xyz_map_in_rgb = transform_points_batch(
            H_NL_RGB, xyz_map_in_new_left.reshape(-1, 3)
        ).reshape(512, 512, 3)
        return xyz_map_in_rgb

    def get_xyz_map_in_world(self, frame_id: int):
        """
        Method to get the xyz map in the world frame for the given frame id.
        """

        # Get the xyz map in the new left frame
        xyz_map_in_new_left = self.get_xyz_map_in_new_left(frame_id)

        # Get the new_left to world transform
        device_to_world = self.vrs_demo.get_device_to_world(frame_id)
        left_to_device = self.vrs_demo.get_camera_projections()["extrinsics"][
            "slam-front-left"
        ]
        H_L_W = device_to_world @ left_to_device
        H_L_NL = np.eye(4)
        H_L_NL[:3, :3] = self.R_L_NL
        H_NL_W = H_L_W @ H_L_NL

        # Transform the xyz map to the world frame
        xyz_map_in_world = transform_points_batch(
            H_NL_W, xyz_map_in_new_left.reshape(-1, 3)
        ).reshape(512, 512, 3)

        return xyz_map_in_world
