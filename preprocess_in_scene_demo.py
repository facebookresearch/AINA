# Copyright (c) Meta Platforms, Inc. and affiliates.

import os

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from tqdm import tqdm

from aina.preprocessing.in_scene.hand_pose_detector import HandPoseDetectionWrapper
from aina.preprocessing.point_utils.grounded_segmentation import GroundedSAMWrapper
from aina.preprocessing.point_utils.point_tracker import CoTrackerPoints
from aina.robot.camera_api import CameraAPI
from aina.utils.constants import HAMER_PATH
from aina.utils.points import (
    get_pcd_from_depth_image,
    transform_points_batch,
    triangulate_points,
)
from aina.utils.rerun_visualizer import RerunVisualizer
from aina.utils.video_recorder import VideoRecorder


def convert_video_to_images(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    images = []
    pbar = tqdm(total=frame_count)
    while success:
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        success, image = vidcap.read()
        pbar.update(1)

    pbar.close()

    return np.array(images)


class InSceneDemo:
    def __init__(self, triangulate_hand_pose=False):

        # Initialize the grounded segmentation model
        self.triangulate_hand_pose = triangulate_hand_pose

    def _load_data(self, demo_dir):
        self.left_images = convert_video_to_images(os.path.join(demo_dir, "left.mp4"))
        self.right_images = convert_video_to_images(os.path.join(demo_dir, "right.mp4"))
        self.left_depth_images = np.load(
            os.path.join(demo_dir, "left_depth_images.npy")
        )
        self.right_depth_images = np.load(
            os.path.join(demo_dir, "right_depth_images.npy")
        )

        self.demo_dir = demo_dir

    def get_initial_tracking_points(self, text_prompts):

        print(f"self.left_images[0].shape: {self.left_images[0].shape}")
        grounded_sam = GroundedSAMWrapper(grid_size=5, hugging_face=True)
        left_image = self.left_images[0]
        right_image = self.right_images[0]
        left_points_to_track = grounded_sam.get_segmented_points(
            Image.fromarray(self.left_images[0]),
            text_prompts,
            visualize=True,
            max_points=None,
            image_name=f"{self.demo_dir}/left_image_segmented_points",
        )
        left_points_to_track = np.concatenate(left_points_to_track, axis=0)
        right_points_to_track = grounded_sam.get_segmented_points(
            Image.fromarray(self.right_images[0]),
            text_prompts,  # ["yellow cup", "black cylinder"],
            visualize=True,
            max_points=None,
            image_name=f"{self.demo_dir}/right_image_segmented_points",
        )
        right_points_to_track = np.concatenate(right_points_to_track, axis=0)

        grounded_sam.to("cpu")
        # breakpoint()
        torch.cuda.empty_cache()
        return left_points_to_track, right_points_to_track

    def get_demo_points(self, left_points_to_track, right_points_to_track):
        # Initialize point tracker
        cotracker_wrapper = CoTrackerPoints(device="cuda", is_online=True)
        left_points = cotracker_wrapper.track_points_from_frames(
            self.left_images, left_points_to_track, return_numpy=True
        )

        right_points = cotracker_wrapper.track_points_from_frames(
            self.right_images, right_points_to_track, return_numpy=True
        )

        # print(f"Left points: {self.left_points.shape}")
        # print(f"Right points: {self.right_points.shape}")

        cotracker_wrapper.to("cpu")
        # breakpoint()
        torch.cuda.empty_cache()
        return left_points, right_points

    def _get_camera_pcd(self, t, camera_name, camera_api, points):
        if camera_name == "left":
            depth_image = self.left_depth_images[t]
            depth_image = depth_image.squeeze() / 1000.0
            # depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)
            rgb_image = self.left_images[t]
            points_2d = points[t]

        elif camera_name == "right":
            depth_image = self.right_depth_images[t]
            depth_image = depth_image.squeeze() / 1000.0
            rgb_image = self.right_images[t]
            points_2d = points[t]

        pcd = get_pcd_from_depth_image(
            rgb_image=rgb_image,
            depth_image=depth_image,
            intrinsic_matrix=camera_api.get_intrinsics(),
            points_2d=points_2d.astype(np.int16),
            sample=False,
        )

        # Transform the points to the base frame
        points = transform_points_batch(
            camera_api.get_extrinsics(),
            pcd.points,
        )
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def _get_triangulated_hand_pose(
        self, left_api, right_api, left_points, right_points
    ):

        # left_points = left_points[t]
        # right_points = right_points[t]

        assert left_points.shape == (21, 2), f"Left points: {left_points.shape}"
        assert right_points.shape == (21, 2), f"Right points: {right_points.shape}"

        left_projection = left_api.get_projection_matrix()
        right_projection = right_api.get_projection_matrix()

        left_points = triangulate_points(
            [left_projection, right_projection], [left_points, right_points]
        )

        return left_points[:, :3]

    def get_hand_pose(self, image, camera_api):
        # breakpoint()
        fingertips, hamer_frame, wrist_pose, fingertips_2d = (
            self.hamer_wrapper.get_hand_pose(
                image,
                render=False,
                is_right_hand=True,
                focal_length=torch.tensor(camera_api.get_focal_length(), device="cuda"),
            )
        )

        if fingertips is not None and wrist_pose is not None:
            fingertips = transform_points_batch(camera_api.get_extrinsics(), fingertips)
            wrist_pose = transform_points_batch(
                camera_api.get_extrinsics(), wrist_pose.reshape(1, 3)
            )
            fingertips = fingertips - fingertips[0] + wrist_pose

        return fingertips, hamer_frame, wrist_pose, fingertips_2d

    def run(self, demo_dir, text_prompts, visualize=False, dump_visualizations=False):

        # Move all the points to the base frame and visualize them
        # visualizer = RerunVisualizer(window_name="On Scene Demo", rerun_type="local")
        if visualize:
            self.rerun_tool = RerunVisualizer(
                window_name="On Scene Demo", rerun_type="local"
            )

        if dump_visualizations:
            left_video_recorder = VideoRecorder(
                save_dir=f"{demo_dir}",
                fps=15,
            )
            right_video_recorder = VideoRecorder(
                save_dir=f"{demo_dir}",
                fps=15,
            )

        self._load_data(demo_dir)
        left_points_to_track, right_points_to_track = self.get_initial_tracking_points(
            text_prompts
        )
        left_points, right_points = self.get_demo_points(
            left_points_to_track, right_points_to_track
        )

        left_camera_api = CameraAPI("left")
        right_camera_api = CameraAPI("right")
        self.hamer_wrapper = HandPoseDetectionWrapper(hamer_path=HAMER_PATH)

        hand_poses = []
        object_poses = []

        min_num_frames = min(
            len(self.left_images),
            len(self.right_images),
            len(self.left_depth_images),
            len(self.right_depth_images),
        )
        pbar = tqdm(total=min_num_frames)
        for t in range(min_num_frames):
            pbar.set_description(f"Getting hand pose + pcd - {t}")

            left_pcd = self._get_camera_pcd(t, "left", left_camera_api, left_points)
            right_pcd = self._get_camera_pcd(t, "right", right_camera_api, right_points)

            if visualize:
                self.rerun_tool.add_pcd(
                    pcd=left_pcd.points,
                    point_colors=np.asarray(left_pcd.colors),
                    time=t,
                    name_tag="pcd/left",
                    radius=0.005,
                )
                self.rerun_tool.add_pcd(
                    pcd=right_pcd.points,
                    point_colors=np.asarray(right_pcd.colors),
                    time=t,
                    name_tag="pcd/right",
                    radius=0.005,
                )

            torch.cuda.empty_cache()
            (
                hand_pose_in_left,
                left_hamer_image,
                wrist_pose_in_left,
                hand_pose_2d_left,
            ) = self.get_hand_pose(self.left_images[t], left_camera_api)
            if hand_pose_in_left is not None and visualize:
                self.rerun_tool.add_pcd(
                    pcd=hand_pose_in_left,
                    point_colors=(255, 0, 0),
                    time=t,
                    name_tag="pcd/hand_pose_left",
                    radius=0.005,
                )
                self.rerun_tool.add_pcd(
                    pcd=wrist_pose_in_left,
                    point_colors=(255, 0, 0),
                    time=t,
                    name_tag="pcd/wrist_pose_left",
                    radius=0.01,
                )
                self.rerun_tool.add_image(
                    img=cv2.cvtColor(left_hamer_image, cv2.COLOR_BGR2RGB),
                    time=t,
                    name_tag="left_image",
                )
                self.rerun_tool.add_2d_points(
                    points=hand_pose_2d_left,
                    time=t,
                    name_tag="left_image/hand_keypoints",
                    colors=(255, 0, 0),
                    radius=5,
                )

            (
                hand_pose_in_right,
                right_hamer_image,
                wrist_pose_in_right,
                hand_pose_2d_right,
            ) = self.get_hand_pose(self.right_images[t], right_camera_api)
            if hand_pose_in_right is not None and visualize:
                self.rerun_tool.add_pcd(
                    pcd=hand_pose_in_right,
                    point_colors=(0, 0, 255),
                    time=t,
                    name_tag="pcd/hand_pose_right",
                    radius=0.005,
                )
                self.rerun_tool.add_pcd(
                    pcd=wrist_pose_in_right,
                    point_colors=(0, 0, 255),
                    time=t,
                    name_tag="pcd/wrist_pose_right",
                    radius=0.01,
                )
                self.rerun_tool.add_image(
                    img=cv2.cvtColor(right_hamer_image, cv2.COLOR_BGR2RGB),
                    time=t,
                    name_tag="right_image",
                )
                self.rerun_tool.add_2d_points(
                    points=hand_pose_2d_right,
                    time=t,
                    name_tag="right_image/hand_keypoints",
                    colors=(0, 0, 255),
                    radius=5,
                )

            if hand_pose_in_left is not None and hand_pose_in_right is not None:
                # Get the average of the hand pose in the left and right
                hand_pose_average = np.mean(
                    [hand_pose_in_left, hand_pose_in_right], axis=0
                )
                # Triangulate the hand pose and fit it to the same pcd
                hand_pose_triangulated = self._get_triangulated_hand_pose(
                    left_camera_api,
                    right_camera_api,
                    hand_pose_2d_left,
                    hand_pose_2d_right,
                )
                if self.triangulate_hand_pose:
                    hand_poses.append(hand_pose_triangulated)
                else:
                    # NOTE: We'll only include frames where both hands are visible
                    hand_poses.append(hand_pose_average)

                left_pcd = np.concatenate([left_pcd.points, left_pcd.colors], axis=-1)
                right_pcd = np.concatenate(
                    [right_pcd.points, right_pcd.colors], axis=-1
                )
                object_poses.append(np.concatenate([left_pcd, right_pcd], axis=0))

                if visualize:
                    self.rerun_tool.add_pcd(
                        pcd=hand_pose_average,
                        point_colors=(0, 255, 0),
                        time=t,
                        name_tag="pcd/hand_pose_average",
                    )

                    self.rerun_tool.add_pcd(
                        pcd=hand_pose_triangulated,
                        point_colors=(0, 255, 255),
                        time=t,
                        name_tag="pcd/triangulated_hand_pose",
                    )

                if dump_visualizations:
                    left_image = self.left_images[t]
                    right_image = self.right_images[t]

                    # Plot object points on the images
                    for point in left_points[t]:
                        left_image = cv2.circle(
                            left_image, tuple(point.astype(int)), 1, (255, 0, 0), -1
                        )
                    for point in right_points[t]:
                        right_image = cv2.circle(
                            right_image, tuple(point.astype(int)), 1, (255, 0, 0), -1
                        )

                    # Plot hand pose on the images
                    for point in hand_pose_2d_left:
                        left_image = cv2.circle(
                            left_image, tuple(point.astype(int)), 3, (0, 255, 255), -1
                        )
                    for point in hand_pose_2d_right:
                        right_image = cv2.circle(
                            right_image, tuple(point.astype(int)), 3, (0, 255, 255), -1
                        )

                    left_video_recorder.record(left_image)
                    right_video_recorder.record(right_image)

            pbar.update(1)

        pbar.close()
        object_poses = np.array(object_poses)
        hand_poses = np.array(hand_poses)

        print(f"Object poses: {object_poses.shape}")
        print(f"Hand poses: {hand_poses.shape}")

        np.save(os.path.join(self.demo_dir, "object-poses-in-base.npy"), object_poses)
        np.save(os.path.join(self.demo_dir, "hand-poses-in-base.npy"), hand_poses)

        if dump_visualizations:
            left_video_recorder.save(f"left_visualization.mp4")
            right_video_recorder.save(f"right_visualization.mp4")

        return object_poses, hand_poses


if __name__ == "__main__":

    from aina.utils.file_ops import get_repo_root

    demo_dir = os.path.join(
        get_repo_root(),
        "data/osfstorage/human_data",
    )
    in_scene_demo = InSceneDemo(triangulate_hand_pose=True)
    in_scene_demo.run(
        demo_dir,
        text_prompts=["bowl", "toaster oven"],
        visualize=True,
        dump_visualizations=True,
    )
