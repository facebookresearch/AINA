# Script to save videos from cameras

import argparse
import os

import numpy as np

from aina.robot.camera_api import CameraAPI
from aina.utils.timer import FrequencyTimer
from aina.utils.video_recorder import VideoRecorder


def save_video_from_cameras(camera_names, save_dir, frequency=10):
    camera_apis = {}
    video_recorders = {}
    depth_images = {}
    for camera_name in camera_names:
        camera = CameraAPI(camera_name)
        camera_apis[camera_name] = camera
        depth_images[camera_name] = []

    os.makedirs(save_dir, exist_ok=True)
    for camera_name in camera_names:
        video_recorders[camera_name] = VideoRecorder(save_dir, fps=frequency)

    timer = FrequencyTimer(frequency)

    input("Press Enter to start recording...")

    try:
        while True:
            timer.start_loop()

            for camera_name, camera in camera_apis.items():
                image, _ = camera.get_rgb_image()
                depth, _ = camera.get_depth_image()
                video_recorders[camera_name].record(image)
                depth_images[camera_name].append(depth)

            timer.end_loop()

    except KeyboardInterrupt:
        print("Recording stopped")
    finally:
        print("Stopping recording...")

        print("Saving videos...")
        for camera_name, video_recorder in video_recorders.items():
            video_recorder.save(f"{camera_name}.mp4")

            depth_image = np.array(depth_images[camera_name])
            np.save(
                os.path.join(save_dir, f"{camera_name}_depth_images.npy"),
                depth_image,
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_num", type=str, default="5")
    parser.add_argument("--task_name", type=str, default="toy_picking_small")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo_num = args.demo_num
    task_name = args.task_name
    os.makedirs(f"./human_data/{task_name}", exist_ok=True)
    save_video_from_cameras(
        ["left", "right"],
        f"./human_data/{task_name}/demo_{demo_num}",
    )
