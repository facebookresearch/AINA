# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from termcolor import cprint

from aina.utils.points import rigid_transform


# Aligns an Aria demo to in-scene demo
class DemoAligner:
    def __init__(
        self,
        vrs_demo,
        object_tracker,
        in_scene_demo_dir,
        use_stable_points=False,
    ):
        # In-the-wild segmented demo and object tracker and On-scene demo directory
        self.vrs_demo = vrs_demo
        self.object_tracker = object_tracker
        self.in_scene_demo_dir = in_scene_demo_dir
        self.use_stable_points = use_stable_points

        self.in_scene_object_points = np.load(
            f"{in_scene_demo_dir}/object-poses-in-base.npy"
        )[
            :, :, :3
        ]  # Remove the color channel

        self.in_scene_hand_points = np.load(
            f"{in_scene_demo_dir}/hand-poses-in-base.npy"
        )[
            :, :
        ]  # Remove the color channel

    def convert_hand_points_order(self, aria_hand_points):
        aria_finger_ids = {  # keypoints from tip to knuckls and the palm
            "thumb": [0, 7, 6, 20, 5],  # 20: palm, 5: wrist
            "index": [1, 10, 9, 8],
            "middle": [2, 13, 12, 11],
            "ring": [3, 16, 15, 14],
            "pinky": [4, 19, 18, 17],
        }
        on_scene_finger_ids = {  # keypoints from tip to knuckles
            "thumb": [4, 3, 2, 1, 0],
            "index": [8, 7, 6, 5],
            "middle": [12, 11, 10, 9],
            "ring": [16, 15, 14, 13],
            "pinky": [20, 19, 18, 17],
        }

        transformed_hand_points = np.zeros_like(aria_hand_points)
        for finger in aria_finger_ids.keys():
            for i in range(len(aria_finger_ids[finger])):
                if aria_hand_points.ndim == 3:
                    transformed_hand_points[:, on_scene_finger_ids[finger][i], :] = (
                        aria_hand_points[:, aria_finger_ids[finger][i], :]
                    )
                else:
                    transformed_hand_points[on_scene_finger_ids[finger][i], :] = (
                        aria_hand_points[aria_finger_ids[finger][i], :]
                    )

        return transformed_hand_points

    def _get_center_of_mass_distance(self, rotated_object_points_in_world):
        if self.use_stable_points:
            stable_on_scene_object_points = self.find_stable_points(
                self.in_scene_object_points
            )[0]
            stable_aria_object_points = self.find_stable_points(
                rotated_object_points_in_world
            )[0]

            in_scene_object_points_com = np.mean(
                stable_on_scene_object_points,
                axis=0,
            )
            aria_object_points_com = np.mean(
                stable_aria_object_points,
                axis=0,
            )
        else:

            in_scene_object_points_com = np.mean(
                self.in_scene_object_points[0],
                axis=0,
            )
            aria_object_points_com = np.mean(
                rotated_object_points_in_world[0],
                axis=0,
            )
        com_distance = aria_object_points_com - in_scene_object_points_com
        return com_distance

    def get_all_rotated_object_points_in_world(self, visualizer):
        # This method will rotate the object and hand points such that the hand would look in the same direction as the on scene hand
        object_points_in_base = []
        hand_points_in_base = []

        _, points_3d = self.object_tracker.get_demo_points()

        for demo_frame_id in range(points_3d.shape[0]):
            # Get the object points in the world frame
            device_to_world = self.vrs_demo.get_device_to_world(demo_frame_id)
            object_points_in_world = points_3d[
                demo_frame_id
            ]  # With the new implementation points-3d are saved with respect to the world

            # Get the hand points in the world frame and convert the order
            hand_points_in_world = self.vrs_demo.get_hand_poses_in_frame(
                demo_frame_id, "world"
            )
            hand_points_in_world = self.convert_hand_points_order(
                hand_points_in_world["right"]  # We're only looking at right hand
            )

            # Calculate the COM of each object point and shift the points
            if (
                demo_frame_id == 0
            ):  # Shift the aria object points to the on scene object points

                initial_hand_to_on_scene_hand_rotation, _ = rigid_transform(
                    hand_points_in_world, self.in_scene_hand_points[0]
                )
                # Convert to scipy rotation
                rot = R_scipy.from_matrix(initial_hand_to_on_scene_hand_rotation)

                # Extract Euler angles (ZYX order: first Z, then Y, then X)
                z, _, _ = rot.as_euler("zyx", degrees=False)
                init_hand_rotation = R_scipy.from_euler(
                    "z", z
                ).as_matrix()  # 3x3 matrix

                object_points_in_world = object_points_in_world @ init_hand_rotation.T
                hand_points_in_world = hand_points_in_world @ init_hand_rotation.T

            else:
                # Rotate the object and hand points by the rotation matrix
                hand_points_in_world = hand_points_in_world @ init_hand_rotation.T
                object_points_in_world = object_points_in_world @ init_hand_rotation.T

            object_points_in_base.append(object_points_in_world)
            hand_points_in_base.append(hand_points_in_world)

        object_points_in_base = np.stack(object_points_in_base)
        hand_points_in_base = np.stack(hand_points_in_base)

        return object_points_in_base, hand_points_in_base

    def find_stable_points(self, object_points_in_base):
        # Object points are in the base frame (T, N, 3)
        # Find the objects that don't move
        # Calculate the mean of the object points
        object_points_com = np.mean(object_points_in_base, axis=0)

        # Calculate the distance between the object points and the object points com
        distance = np.linalg.norm(
            object_points_in_base - object_points_com, axis=(0, 2)
        )

        # Find the objects that don't move
        threshold = 0.5
        objects_that_dont_move = object_points_in_base[:, distance < threshold, :]
        while objects_that_dont_move.shape[1] == 0:
            threshold += 0.05
            objects_that_dont_move = object_points_in_base[:, distance < threshold, :]
        return objects_that_dont_move

    def align_demo(self, visualize=False):

        cprint(
            f"[DemoAligner] Aligning demo",
            "cyan",
        )

        if visualize:
            from aina.utils.rerun_visualizer import RerunVisualizer

            visualizer = RerunVisualizer(
                window_name=f"Demo Aligner",
                rerun_type="local",
            )

        # Rotate the aria object and hand points
        all_rotated_object_points_in_world, all_rotated_hand_points_in_world = (
            self.get_all_rotated_object_points_in_world(visualizer)
        )

        # Shift the aria object and hand points to the on scene object points
        com_distance = self._get_center_of_mass_distance(
            all_rotated_object_points_in_world,
        )
        all_rotated_object_points_in_world = (
            all_rotated_object_points_in_world - com_distance
        )
        all_rotated_hand_points_in_world = (
            all_rotated_hand_points_in_world - com_distance
        )

        demo_root = self.vrs_demo.get_demo_root()
        cprint(
            f"[DemoAligner] Saving the object {all_rotated_object_points_in_world.shape} and hand points {all_rotated_hand_points_in_world.shape} to {demo_root}",
            "cyan",
        )
        # Save the points
        np.save(
            f"{demo_root}/object-poses-in-base.npy",
            all_rotated_object_points_in_world,
        )
        np.save(
            f"{demo_root}/hand-poses-in-base.npy",
            all_rotated_hand_points_in_world,
        )
        cprint(
            f"[DemoAligner] ################ Demo: {demo_root} Processed ################",
            "cyan",
        )

        # Visualize everything
        if visualize:
            for demo_frame_id in range(all_rotated_object_points_in_world.shape[0]):
                visualizer.add_pcd(
                    pcd=all_rotated_object_points_in_world[demo_frame_id],
                    name_tag=f"object_points_in_world",
                    time=demo_frame_id,
                    radius=0.003,
                    point_colors=(255, 255, 0),
                )
                visualizer.add_pcd(
                    pcd=all_rotated_hand_points_in_world[demo_frame_id],
                    name_tag=f"hand_points_in_world",
                    time=demo_frame_id,
                    radius=0.005,
                    point_colors=(0, 255, 0),
                )
                visualizer.add_pcd(
                    pcd=(
                        self.in_scene_object_points[demo_frame_id]
                        if demo_frame_id < len(self.in_scene_object_points)
                        else self.in_scene_object_points[-1]
                    ),
                    name_tag=f"on_scene_object_points",
                    time=demo_frame_id,
                    radius=0.003,
                    point_colors=(0, 255, 255),
                )
                visualizer.add_pcd(
                    pcd=(
                        self.in_scene_hand_points[demo_frame_id]
                        if demo_frame_id < len(self.in_scene_hand_points)
                        else self.in_scene_hand_points[-1]
                    ),
                    name_tag=f"on_scene_hand_points",
                    time=demo_frame_id,
                    radius=0.005,
                    point_colors=(0, 0, 255),
                )
