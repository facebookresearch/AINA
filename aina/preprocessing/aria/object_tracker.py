# This file will input a vrs demo, depth extractor, demo frame ids and a text prompt
# It will track the objects in 2d and 3d

import os

import numpy as np
from PIL import Image
from termcolor import cprint
from tqdm import tqdm

from aina.preprocessing.point_utils.grounded_segmentation import GroundedSAMWrapper
from aina.preprocessing.point_utils.point_tracker import CoTrackerPoints
from aina.utils.file_ops import suppress
from aina.utils.points import (
    get_points_3d_in_rgb_from_xyz_maps,
    to_open3d_pcd,
    transform_points_batch,
)


class ObjectTracker:
    def __init__(self, vrs_demo, depth_extractor, text_prompt):
        self.vrs_demo = vrs_demo
        self.depth_extractor = depth_extractor
        self.text_prompts = text_prompt

    def get_object_queries(self, rgb_image, text_prompts, demo_root):
        # with suppress(stdout=True):
        # Initialize the grounded sam
        cprint("Getting object queries", "magenta")
        grounded_sam = GroundedSAMWrapper(grid_size=4, device="cuda", hugging_face=True)

        # cv2.imwrite("queries_img.png", img=cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        chosen_points = grounded_sam.get_segmented_points(
            image=Image.fromarray(rgb_image),
            text=text_prompts,
            visualize=True,
            image_name=f"{demo_root}/object_queries",
        )
        # breakpoint()
        point_counter = [cp.shape[0] for cp in chosen_points]
        if len(chosen_points) == len(text_prompts):
            cprint("Got all object queries", "magenta")
        else:
            cprint(
                f"Got {len(chosen_points)} object queries, expected {len(text_prompts)}",
                "red",
            )
            # raise Exception("Got wrong number of object queries")
        chosen_points = np.concatenate(chosen_points, axis=0)
        grounded_sam.to("cpu")
        cprint("Got object queries", "magenta")

        return chosen_points, point_counter

    def _get_2d_points(self, chosen_points, rgb_images):
        # Initialize the cotracker
        cprint("Initializing cotracker", "magenta")
        cotracker_wrapper = CoTrackerPoints(device="cuda", is_online=True)

        points_2d, _ = cotracker_wrapper.track_points_from_frames(
            rgb_images,
            chosen_points,
            return_numpy=True,
            return_visibility=True,
        )

        cotracker_wrapper.to("cpu")

        return points_2d

    def _get_3d_points(self, points_2d):
        # This will use the depth extractor to unproject the 2D points to 3D
        # Traverse through the demo frame ids and get xyz map in rgb frame
        all_points_3d = []
        for frame_idx in range(points_2d.shape[0]):
            xyz_map_in_rgb = self.depth_extractor.get_xyz_map_in_rgb(frame_idx)
            points_3d, _ = get_points_3d_in_rgb_from_xyz_maps(
                xyz_map_in_rgb,
                points_2d_in_rgb=points_2d[frame_idx],
                rgb_intrinsics=self.vrs_demo.get_camera_projections()["intrinsics"][
                    "camera-rgb"
                ],
            )
            # breakpoint()
            device_to_world = self.vrs_demo.get_device_to_world(frame_idx)
            rgb_to_device = self.vrs_demo.get_camera_projections()["extrinsics"][
                "camera-rgb"
            ]
            rgb_to_world = device_to_world @ rgb_to_device
            points_in_world = transform_points_batch(
                current_to_target_transform=rgb_to_world, points=points_3d
            )
            all_points_3d.append(points_in_world)

        all_points_3d = np.stack(all_points_3d)
        return all_points_3d

    def get_demo_points(self, visualize=False):

        # Check whether points 2d or points 3d exists
        demo_root = self.vrs_demo.get_demo_root()

        # Get the demo timestamps
        cprint(f"[ObjectTracker] Tracking objects for demo", "magenta")

        # Get the object queries
        if not os.path.exists(os.path.join(demo_root, "chosen-points.npy")):
            with suppress(stdout=True):
                chosen_points, _ = self.get_object_queries(
                    rgb_image=self.vrs_demo.get_rgb_image(0),
                    text_prompts=self.text_prompts,
                    demo_root=demo_root,
                )
            np.save(os.path.join(demo_root, "chosen-points.npy"), chosen_points)
        else:
            chosen_points = np.load(os.path.join(demo_root, "chosen-points.npy"))
        cprint(f"[ObjectTracker] Tracking {chosen_points.shape[0]} points", "magenta")

        # Get the 2D points
        if not os.path.exists(os.path.join(demo_root, "points-2d.npy")):
            demo_data_dict = self.vrs_demo.get_demo_data_dict()
            points_2d = self._get_2d_points(
                chosen_points,
                demo_data_dict["undistorted-camera-rgb"][:],
            )
            np.save(os.path.join(demo_root, "points-2d.npy"), points_2d)
        else:
            points_2d = np.load(os.path.join(demo_root, "points-2d.npy"))
        cprint(f"[ObjectTracker] Dumped 2D {points_2d.shape} points", "magenta")

        # Get the 3D points
        if not os.path.exists(os.path.join(demo_root, "points-3d.npy")):
            points_3d = self._get_3d_points(points_2d)
            np.save(os.path.join(demo_root, "points-3d.npy"), points_3d)
        else:
            points_3d = np.load(os.path.join(demo_root, "points-3d.npy"))
        cprint(f"[ObjectTracker] Dumped 3D {points_3d.shape} points", "magenta")

        if visualize:
            from aina.utils.rerun_visualizer import RerunVisualizer

            rerun_visualizer = RerunVisualizer(
                window_name=f"Object Points Visualization", rerun_type="local"
            )

            pbar = tqdm(range(0, points_2d.shape[0]))
            for i, demo_frame_id in enumerate(pbar):
                # Visualize 3d points in world frame

                xyz_map = self.depth_extractor.get_xyz_map_in_world(demo_frame_id)
                rectified_slam_frames = self.depth_extractor.get_rectified_slam_frames(
                    demo_frame_id
                )
                rectified_left = rectified_slam_frames["left"]

                if len(rectified_left.shape) < len(xyz_map.shape):
                    rectified_left = np.repeat(
                        np.expand_dims(rectified_left, -1), 3, -1
                    )
                pcd = to_open3d_pcd(
                    xyz_map.reshape(-1, 3),
                    rectified_left.reshape(
                        -1, 3
                    ),  # NOTE: This is the image that we are using to visualize the points
                )

                hand_poses_in_world = self.vrs_demo.get_hand_poses_in_frame(
                    demo_frame_id, "world"
                )
                left_hand_pose = hand_poses_in_world["left"]
                right_hand_pose = hand_poses_in_world["right"]

                rerun_visualizer.add_pcd(
                    pcd=np.asarray(pcd.points),
                    name_tag="pcd",
                    point_colors=np.asarray(pcd.colors),
                    radius=0.002,
                    time=i,
                )
                rerun_visualizer.add_pcd(
                    pcd=left_hand_pose,
                    name_tag="left_hand_pose",
                    color=(0, 200, 200),
                    radius=0.01,
                    time=i,
                )
                rerun_visualizer.add_pcd(
                    pcd=right_hand_pose,
                    name_tag="right_hand_pose",
                    color=(0, 200, 200),
                    radius=0.01,
                    time=i,
                )

                # Add object points to this
                points_3d_in_world = points_3d[i]
                rerun_visualizer.add_pcd(
                    pcd=points_3d_in_world,
                    name_tag="object_points",
                    color=(200, 0, 0),
                    radius=0.01,
                    time=i,
                )

                # Add 2d points to this
                points_2d_in_rgb = points_2d[i]
                rgb_image = self.vrs_demo.get_rgb_image(demo_frame_id)
                rerun_visualizer.add_image(
                    img=rgb_image,
                    name_tag=f"rgb",
                    time=i,
                )
                rerun_visualizer.add_2d_points(
                    points=points_2d_in_rgb,
                    name_tag=f"rgb/points_2d",
                    radius=0.5,
                    colors=(255, 0, 0),
                    time=i,
                )

                pbar.update(1)
                pbar.set_description(
                    f"[ObjectTracker] Visualizing frame {demo_frame_id}"
                )

        return points_2d, points_3d
