# Copyright (c) Meta Platforms, Inc. and affiliates.

import cv2
import cv2.aruco as aruco
import numpy as np


def project_axes(rvec, tvec, intrinsic_matrix, scale=0.01, dist=None):
    """
    Draw a 6dof axis (XYZ -> RGB) in the given rotation and translation
    :param img - rgb numpy array
    :rvec - rotation matrix
    :t - 3d translation vector, in meters (dtype must be float)
    :K - intrinsic calibration matrix , 3x3
    :scale - factor to control the axis lengths
    :dist - optional distortion coefficients, numpy array of length 4. If None distortion is ignored.
    """
    # img = img.astype(np.float32)
    dist = np.zeros(4, dtype=float) if dist is None else dist
    points = scale * np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(
        -1, 3
    )
    axis_points, _ = cv2.projectPoints(points, rvec, tvec, intrinsic_matrix, dist)
    return axis_points


def project_poses(poses, intrinsic_matrix, distortion=None, scale=0.01):
    # Project the axes for each pose
    projected_poses = []
    # print('poses.shape: {}'.format(poses.shape))
    for pose_id in range(len(poses)):
        pose = poses[pose_id]
        rvec, tvec = pose[:3, :3], pose[:3, 3]
        projected_pose = project_axes(
            rvec, tvec, intrinsic_matrix, dist=distortion, scale=scale
        )
        projected_poses.append(projected_pose)
    projected_poses = np.stack(projected_poses, axis=0)

    return projected_poses


def plot_points(points, img, colors, radius=3):
    for i, point in enumerate(points):
        point = point.astype(int)
        img = cv2.circle(img, tuple(point.ravel()), radius, colors[i], -1)
    return img
