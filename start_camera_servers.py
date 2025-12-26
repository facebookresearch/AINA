# Copyright (c) Meta Platforms, Inc. and affiliates.

import multiprocessing as mp
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from aina.utils.constants import *
from aina.utils.zmq import ZMQCameraPublisher

np.set_printoptions(precision=3, suppress=True)


class RealsenseROSAPI(Node):
    def __init__(self, camera_name):
        super().__init__(f"{camera_name}_realsense_ros_api")
        self.rgb_subscriber = self.create_subscription(
            Image,
            f"{REALSENSE_TOPIC_PREFIXES[camera_name]}/color/image_raw",
            self.listener_callback_rgb,
            10,
        )
        self.camera_name = camera_name
        self.rgb_subscriber

        self.depth_subscriber = self.create_subscription(
            Image,
            f"{REALSENSE_TOPIC_PREFIXES[camera_name]}/aligned_depth_to_color/image_raw",
            self.listener_callback_depth,
            10,
        )
        self.depth_subscriber

        # Start the ZMQ Publishers
        self.publisher = ZMQCameraPublisher(
            host="localhost", port=CAMERA_PORTS[camera_name]
        )
        self.depth_publisher = ZMQCameraPublisher(
            host="localhost", port=CAMERA_DEPTH_PORTS[camera_name]
        )

    def listener_callback_rgb(self, msg):
        image = np.array(msg.data).reshape(msg.height, msg.width, 3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.publisher.pub_rgb_image(image, time.time())

    def listener_callback_depth(self, msg):
        image = np.frombuffer(msg.data, dtype=np.int16).reshape(
            msg.height, msg.width, 1
        )
        self.depth_publisher.pub_depth_image(image, time.time())


def start_realsense_ros_api(camera_name):
    rclpy.init()
    realsense_ros_api = RealsenseROSAPI(camera_name)
    rclpy.spin(realsense_ros_api)
    realsense_ros_api.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":

    processes = [
        mp.Process(
            target=start_realsense_ros_api,
            args=("left",),
        ),
        mp.Process(
            target=start_realsense_ros_api,
            args=("right",),
        ),
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()
