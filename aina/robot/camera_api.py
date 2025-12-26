# Copyright (c) Meta Platforms, Inc. and affiliates.

import cv2
import numpy as np

from aina.utils.constants import *
from aina.utils.zmq import ZMQCameraSubscriber


class CameraAPI:
    def __init__(self, camera_name):
        self.camera_name = camera_name
        self.camera_port = CAMERA_PORTS[camera_name]
        self.camera_depth_port = CAMERA_DEPTH_PORTS[camera_name]

        self.subscriber = ZMQCameraSubscriber(
            host="localhost", port=self.camera_port, topic_type="RGB"
        )
        self.depth_subscriber = ZMQCameraSubscriber(
            host="localhost", port=self.camera_depth_port, topic_type="Depth"
        )

    def get_rgb_image(self):
        rgb_image, timestamp = self.subscriber.recv_rgb_image()
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        return rgb_image, timestamp

    def get_depth_image(self):
        depth_image, timestamp = self.depth_subscriber.recv_depth_image()
        return depth_image, timestamp

    def get_intrinsics(self):
        if self.camera_name == "left":
            return LEFT_INTRINSICS
        elif self.camera_name == "right":
            return RIGHT_INTRINSICS

    def get_distortion_coefficients(self):
        return np.zeros(5)

    def get_extrinsics(self):
        """
        Extrinsics of all the cameras with respect to the base of the robot.
        These transformations are camera-to-base transformations.
        """
        if self.camera_name == "left":
            return LEFT_TO_BASE
        elif self.camera_name == "right":
            return RIGHT_TO_BASE

        return None

    def get_projection_matrix(self):
        intrinsic_matrix = self.get_intrinsics()
        extrinsic_matrix = np.linalg.pinv(self.get_extrinsics())[:3, :]
        projection_matrix = intrinsic_matrix @ extrinsic_matrix
        return projection_matrix

    def get_focal_length(self):

        return self.get_intrinsics()[0, 0]

    def stop(self):
        self.subscriber.stop()


if __name__ == "__main__":
    camera_name = "left"
    camera_api = CameraAPI(camera_name)
    rgb_image, timestamp = camera_api.get_rgb_image()
    cv2.imwrite(
        f"{camera_name}_rgb_image.png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    )
    depth_image, timestamp = camera_api.get_depth_image()
    cv2.imwrite(f"{camera_name}_depth_image.png", depth_image)
    intrinsics = camera_api.get_intrinsics()
    print(f"Intrinsics: {intrinsics}")
    camera_api.stop()
