# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import dill as pickle
import numpy as np
import projectaria_tools.core.calibration as calibration
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId
from termcolor import cprint
from tqdm import tqdm

from aina.utils.file_ops import suppress
from aina.utils.points import transform_points_batch
from aina.utils.vision import get_baseline_between_cameras


@dataclass
class VRSProcessorConfig:
    skip_begin_sec: float = None
    skip_end_sec: float = None
    enabled_streams: List[str] = field(
        default_factory=lambda: [
            "camera-rgb",
            "vio",
            "slam-front-left",
            "slam-front-right",
            "handtracking",
        ]
    )
    subsample_rates: Dict[str, int] = field(default_factory=dict)
    fps: int = 10
    resize_rgb_image: bool = True
    overwrite: bool = False


class VRSDemo:
    DEFAULT_VIO_HIGH_FREQ_SUBSAMPLE_RATE = 50
    VIZ_IMAGE_SUBSAMPLE = 1

    def __init__(self, vrs_file_path: str, config: VRSProcessorConfig):

        self.vrs_path = vrs_file_path
        self.pickle_path = vrs_file_path.replace(".vrs", ".pkl")
        self.demo_root = os.path.dirname(vrs_file_path)
        self.config = config

        # Get the data provider
        self.vrs_data_provider = data_provider.create_vrs_data_provider(vrs_file_path)
        self.device_calibration = self.vrs_data_provider.get_device_calibration()

        # Set deliver options for queued sensor data
        self.deliver_options = (
            self.vrs_data_provider.get_default_deliver_queued_options()
        )
        self.deliver_options.deactivate_stream_all()
        self.enabled_streams = config.enabled_streams

        # Activate streams and set subsample rates
        with suppress(stdout=True):
            self._activate_and_set_streams_by_labels(
                config.enabled_streams, config.subsample_rates
            )

        # Data fps
        self.fps = config.fps

        # Set crop begin and end
        if config.skip_begin_sec is not None:
            self.deliver_options.set_truncate_first_device_time_ns(
                int(config.skip_begin_sec * 1e9)
            )
        if config.skip_end_sec is not None:
            self.deliver_options.set_truncate_last_device_time_ns(
                int(config.skip_end_sec * 1e9)
            )

        self.stream_mappings = {
            "slam-front-left": StreamId("1201-1"),
            "slam-front-right": StreamId("1201-2"),
            "slam-side-left": StreamId("1201-3"),
            "slam-side-right": StreamId("1201-4"),
            "camera-rgb": StreamId("214-1"),
            "handtracking": StreamId("371-1"),
            "vio": StreamId("371-3"),
        }

        # Variables
        self.resize_rgb_image = config.resize_rgb_image
        self.data_dict = {}

        # Load the synced data
        self._add_camera_projections()
        if os.path.exists(self.pickle_path) and not self.config.overwrite:
            cprint(f"[VRSDemo] Loading data from {self.pickle_path}", "green")
            with open(self.pickle_path, "rb") as f:
                self.data_dict = pickle.load(f)
            self._print_data_dict()
        else:
            cprint(f"[VRSDemo] Loading data from {self.vrs_path}", "green")
            self._load_synced_data()
            cprint(f"[VRSDemo] Dumping data to {self.pickle_path}", "green")
            with open(self.pickle_path, "wb") as f:
                pickle.dump(self.data_dict, f)

    def __len__(self):
        return len(self.data_dict["undistorted-camera-rgb"])

    def get_demo_root(self):
        return self.demo_root

    def _activate_and_set_streams_by_labels(
        self, enabled_stream_labels: List[str], subsample_rates: Dict[str, int]
    ):
        """
        Activate selected streams, and set sub_sample rates.
        """
        # For Vio High Frequency stream, by default set a lower sub_sampling rate from 800Hz -> 10Hz
        if "vio_high_frequency" not in subsample_rates:
            subsample_rates["vio_high_frequency"] = (
                self.DEFAULT_VIO_HIGH_FREQ_SUBSAMPLE_RATE
            )
        available_stream_ids = self.deliver_options.get_stream_ids()

        for stream_label in enabled_stream_labels:
            # for each enabled stream label, first find if it is within VRS
            maybe_stream_id = self.vrs_data_provider.get_stream_id_from_label(
                stream_label
            )

            # Then turn it on
            if maybe_stream_id is not None and maybe_stream_id in available_stream_ids:
                self.deliver_options.activate_stream(maybe_stream_id)

                # Set subsample rates
                if stream_label in subsample_rates:
                    self.deliver_options.set_subsample_rate(
                        maybe_stream_id, subsample_rates[stream_label]
                    )

    def _add_camera_projections(self):
        cprint("[VRSDemo] Loading camera projections...", "green")
        # Will add all camera transforms to the data dictionary
        intrinsics_dict = {}  # All the camera intrinsics with corresponding keys
        extrinsics_dict = {}  # All the camera transforms with corresponding keys
        projection_params_dict = (
            {}
        )  # All the projection parameters with corresponding keys

        for camera_label in [
            "camera-rgb",
            "slam-front-left",
            "slam-front-right",
        ]:
            # Get the extrinsics
            camera_calib = (
                self.vrs_data_provider.get_device_calibration().get_camera_calib(
                    camera_label
                )
            )
            linear_camera_calib = calibration.get_linear_camera_calibration(
                int(camera_calib.get_image_size()[0]),
                int(camera_calib.get_image_size()[1]),
                camera_calib.get_focal_lengths()[0],
                camera_label,
                camera_calib.get_transform_device_camera(),
            )  # NOTE: Since we're using undistorted images, we need to use the linear camera calibration
            camera_to_device_transform = (
                camera_calib.get_transform_device_camera().to_matrix()
            )
            extrinsics_dict[camera_label] = camera_to_device_transform

            # Get the intrinsics
            focal_length = linear_camera_calib.get_focal_lengths()
            principal_point = linear_camera_calib.get_principal_point()
            intrinsic_matrix = np.array(
                [
                    [focal_length[0], 0, principal_point[0]],
                    [0, focal_length[1], principal_point[1]],
                    [0, 0, 1],
                ]
            )
            intrinsics_dict[camera_label] = intrinsic_matrix

            # Get the projection parameters
            projection_params_dict[camera_label] = camera_calib.get_projection_params()

            if camera_label == "camera-rgb" and self.resize_rgb_image:
                # Crop the camera-rgb image's intrinsics
                old_intrinsics = intrinsics_dict["camera-rgb"]
                fx, fy, cx, cy = (
                    old_intrinsics[0, 0],
                    old_intrinsics[1, 1],
                    old_intrinsics[0, 2],
                    old_intrinsics[1, 2],
                )
                W_orig, H_orig = 2016, 1512
                W_crop, H_crop = 1512, 1512
                W_new, H_new = 512, 512

                crop_x0 = (W_orig - W_crop) // 2
                crop_y0 = (H_orig - H_crop) // 2
                cx_new = cx - crop_x0
                cy_new = cy - crop_y0
                scale_x = W_new / W_crop
                scale_y = H_new / H_crop

                fx_new = fx * scale_x
                fy_new = fy * scale_y
                cx_new *= scale_x
                cy_new *= scale_y
                intrinsics_dict["camera-rgb"] = np.array(
                    [
                        [fx_new, 0, cx_new],
                        [0, fy_new, cy_new],
                        [0, 0, 1],
                    ]
                )

        self.data_dict["camera-projections"] = {
            "intrinsics": intrinsics_dict,
            "extrinsics": extrinsics_dict,
            "projection_params": projection_params_dict,
        }

    def _undistort_image(self, image, image_label="camera-rgb"):
        camera_calib = self.vrs_data_provider.get_device_calibration().get_camera_calib(
            image_label
        )
        linear_camera_calib = calibration.get_linear_camera_calibration(
            int(camera_calib.get_image_size()[0]),
            int(camera_calib.get_image_size()[1]),
            camera_calib.get_focal_lengths()[0],
            image_label,
            camera_calib.get_transform_device_camera(),
        )
        # rgb image
        undistorted_image = calibration.distort_by_calibration(
            image,
            linear_camera_calib,
            camera_calib,
        )

        return undistorted_image

    def _add_data_to_dict(self, key, value):
        if key in self.data_dict:
            self.data_dict[key].append(value)
        else:
            self.data_dict[key] = [value]

    def _load_synced_data(self, init_image_id=0, final_image_id=None):
        cprint("[VRSDemo] Loading synced data...", "green")
        """
        Method to load the synced data from the vrs file.
        This data will be used to get camera projections, hand poses and images.
        """
        rgb_stream_id = self.stream_mappings["camera-rgb"]
        num_image_data = self.vrs_data_provider.get_num_data(rgb_stream_id)
        if final_image_id is None:
            final_image_id = num_image_data - 1

        time_domain = TimeDomain.DEVICE_TIME  # query data based on DEVICE_TIME
        option = (
            TimeQueryOptions.CLOSEST
        )  # get data whose time [in TimeDomain] is CLOSEST to query time

        # Get the time related information
        time_difference = 1 / self.fps
        time_difference_ns = int(time_difference * 1e9)
        initial_image_data = self.vrs_data_provider.get_image_data_by_index(
            rgb_stream_id, init_image_id
        )
        last_image_data = self.vrs_data_provider.get_image_data_by_index(
            rgb_stream_id, final_image_id
        )
        last_metric_ts = last_image_data[1].capture_timestamp_ns
        init_metric_ts = initial_image_data[1].capture_timestamp_ns
        metric_ts = init_metric_ts

        num_image_data = int((last_metric_ts - init_metric_ts) / time_difference_ns)

        # Query every N samples, and print queried results.
        pbar = tqdm(total=num_image_data, desc="Loading data")
        frame_idx = 0
        for _ in range(init_image_id, final_image_id):
            try:
                metric_ts += time_difference_ns

                if metric_ts > last_metric_ts:
                    break
                image_data = self.vrs_data_provider.get_image_data_by_time_ns(
                    rgb_stream_id, metric_ts, time_domain, option
                )

                image_frame = image_data[0].to_numpy_array()
                distorted_image = self._undistort_image(image_frame)
                # NOTE: There could be a problem here! We are tracking everything in the undistorted image
                # but the extrinsics is with respect to the distorted image, that could be why when unprojected
                # points don't align well. - Continuing as is for now.
                if self.resize_rgb_image:
                    distorted_image = distorted_image[:, 252:1764]
                    distorted_image = cv2.resize(distorted_image, (512, 512))
                self._add_data_to_dict("undistorted-camera-rgb", distorted_image)

                for key in self.enabled_streams:
                    stream_id = self.stream_mappings[key]
                    if key == "camera-rgb":
                        continue

                    if "slam" in key:
                        data = self.vrs_data_provider.get_image_data_by_time_ns(
                            stream_id, metric_ts, time_domain, option
                        )
                        slam_image = data[0].to_numpy_array()
                        self._add_data_to_dict(f"{key}", slam_image)

                    elif "handtracking" in key:
                        hand_pose_data = (
                            self.vrs_data_provider.get_hand_pose_data_by_time_ns(
                                stream_id, metric_ts, time_domain, option
                            )
                        )
                        if hand_pose_data.left_hand is not None:
                            left_hand_pose = np.array(
                                hand_pose_data.left_hand.landmark_positions_device
                            )
                            left_hand_pose = np.reshape(left_hand_pose, -1)
                            self._add_data_to_dict(
                                "left-hand-poses", left_hand_pose.reshape(21, 3)
                            )
                        else:
                            self._add_data_to_dict("left-hand-poses", np.zeros((21, 3)))

                        if hand_pose_data.right_hand is not None:
                            right_hand_pose = np.array(
                                hand_pose_data.right_hand.landmark_positions_device
                            )
                            right_hand_pose = np.reshape(right_hand_pose, -1)
                            self._add_data_to_dict(
                                "right-hand-poses", right_hand_pose.reshape(21, 3)
                            )
                        else:
                            self._add_data_to_dict(
                                "right-hand-poses", np.zeros((21, 3))
                            )

                    elif "vio" in key:
                        # data = data.vio_high_freq_data()
                        data = self.vrs_data_provider.get_vio_high_freq_data_by_time_ns(
                            stream_id, metric_ts, time_domain, option
                        )
                        data = self._get_device_to_world(data)
                        self._add_data_to_dict(key, data)

                pbar.update(1)
                frame_idx += 1

            except KeyboardInterrupt:
                break

        pbar.close()
        self._print_data_dict()

        return self.data_dict

    def _print_data_dict(self):
        cprint("[VRSDemo] Printing data dictionary...", "green")
        for key in self.data_dict.keys():
            if key == "camera-projections":
                cprint(
                    f"Camera Projections for {self.data_dict[key]['intrinsics'].keys()} -> intrinsics: {self.data_dict[key]['intrinsics']['camera-rgb'].shape} | extrinsics: {self.data_dict[key]['extrinsics']['camera-rgb'].shape}, projection_params: {self.data_dict[key]['projection_params']['camera-rgb'].shape}",
                    "green",
                )
            else:
                self.data_dict[key] = np.stack(self.data_dict[key], axis=0)
                cprint(f"key = {key}, shape = {self.data_dict[key].shape}", "green")

    def _get_device_to_world(self, vio_high_freq_data):
        T_World_Device = vio_high_freq_data.transform_odometry_device
        trans = T_World_Device.translation()
        rot = T_World_Device.rotation()

        # Turn this into a 4x4 matrix
        H = np.eye(4)
        H[:3, :3] = rot.to_matrix()
        H[:3, 3] = trans
        return H

    def get_rgb_image(self, frame_id):
        # RGB image is undistorted
        return self.data_dict["undistorted-camera-rgb"][frame_id]

    def get_hand_poses(self, frame_id):
        # Will return the hand poses for the given frame id
        return dict(
            left=self.data_dict["left-hand-poses"][frame_id],
            right=self.data_dict["right-hand-poses"][frame_id],
        )

    def get_hand_poses_in_frame(self, frame_id, frame_name):
        """
        Method to get the hand poses in a given frame.
        frame_name can be:
        - "device": return the hand poses in the device frame
        - "camera-rgb": return the hand poses in the camera-rgb frame
        - "slam-front-left": return the hand poses in the slam-front-left frame
        - "slam-front-right": return the hand poses in the slam-front-right frame
        - "world": return the hand poses in the world frame
        """

        hand_poses_in_device = self.get_hand_poses(frame_id)
        # Calculate the transform from the device to the frame
        if frame_name == "device":
            return hand_poses_in_device

        # Else, it can either be camera-rgb, slam-front-left, slam-front-right or world
        device_to_world = self.get_device_to_world(frame_id)
        left_hand_pose_in_world = transform_points_batch(
            current_to_target_transform=device_to_world,
            points=hand_poses_in_device["left"],
        )
        right_hand_pose_in_world = transform_points_batch(
            current_to_target_transform=device_to_world,
            points=hand_poses_in_device["right"],
        )
        if frame_name == "world":
            return dict(left=left_hand_pose_in_world, right=right_hand_pose_in_world)

        frame_to_world = self.data_dict["camera-projections"]["extrinsics"][frame_name]
        left_hand_pose_in_frame = transform_points_batch(
            current_to_target_transform=np.linalg.pinv(frame_to_world),
            points=left_hand_pose_in_world,
        )
        right_hand_pose_in_frame = transform_points_batch(
            current_to_target_transform=np.linalg.pinv(frame_to_world),
            points=right_hand_pose_in_world,
        )
        return dict(left=left_hand_pose_in_frame, right=right_hand_pose_in_frame)

    def get_slam_frames(self, frame_id):
        # Slam images are distorted
        # Will return the slam frames for the given frame id
        return dict(
            left=self.data_dict["slam-front-left"][frame_id],
            right=self.data_dict["slam-front-right"][frame_id],
        )

    def get_demo_data_dict(self):
        return self.data_dict

    def get_device_to_world(self, frame_id):
        # Will return the vio data for the given frame id
        return self.data_dict["vio"][frame_id]

    def get_camera_projections(self):
        # Will return the camera projections
        return self.data_dict["camera-projections"]

    def update_camera_intrinsics(self, camera_label, new_intrinsics):
        self.data_dict["camera-projections"]["intrinsics"][
            camera_label
        ] = new_intrinsics

    def get_slam_camera_baseline(self):
        left_to_device = self.data_dict["camera-projections"]["extrinsics"][
            "slam-front-left"
        ]
        right_to_device = self.data_dict["camera-projections"]["extrinsics"][
            "slam-front-right"
        ]
        H_R_L = np.linalg.pinv(left_to_device) @ right_to_device
        baseline = get_baseline_between_cameras(H_R_L)
        return baseline
