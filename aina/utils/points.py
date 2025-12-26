import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree


def to_open3d_pcd(points, colors=None, normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


def depth_to_xyz_map(depth: np.ndarray, K, uvs: np.ndarray = None, zmin=0.1):
    invalid_mask = depth < zmin
    H, W = depth.shape[:2]
    if uvs is None:
        vs, us = np.meshgrid(
            np.arange(0, H), np.arange(0, W), sparse=False, indexing="ij"
        )
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:, 0]
        vs = uvs[:, 1]
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    if invalid_mask.any():
        xyz_map[invalid_mask] = 0
    return xyz_map


# Taken from https://github.com/siddhanthaldar/Point-Policy/point_policy/robot_utils/franka/utils.py#L53
def triangulate_points(P, points):
    """
    Triangulate a batch of points from a variable number of camera views.

    Parameters:
    P: list of 3x4 projection matrices for each camera (currently world2camera transform)
    points: list of Nx2 arrays of normalized image coordinates for each camera

    Returns:
    Nx4 array of homogeneous 3D points
    """
    num_views = len(P)
    assert num_views > 1, "At least 2 cameras are required for triangulation"

    num_points = points[0].shape[0]
    A = np.zeros((num_points, num_views * 2, 4))

    for idx in range(num_views):
        # Set up the linear system for each point
        A[:, idx * 2] = points[idx][:, 0, np.newaxis] * P[idx][2] - P[idx][0]
        A[:, idx * 2 + 1] = points[idx][:, 1, np.newaxis] * P[idx][2] - P[idx][1]

    # Solve the system using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[:, -1, :]

    # Normalize the homogeneous coordinates
    X = X / X[:, 3:]

    return X


def sample_set_to_num_points(object_points, num_points, seed=42):
    """
    Method to sample a set of points to a given number of points.
    Parameters:
        object_points: torch.Tensor, shape (B, N, 3)
        num_points: int, number of points to sample
        seed: int, seed for the random number generator
    Returns:
        object_points: torch.Tensor, shape (B, num_points, 3)
    """
    torch.manual_seed(seed)
    object_points = object_points[
        :, torch.randperm(object_points.shape[-2])[:num_points]
    ]

    while object_points.shape[-2] < num_points:
        # Append the same points to the object points until the number of points is equal to the num_object_points
        object_points = torch.cat(
            [
                object_points,
                object_points[:, : num_points - object_points.shape[-2]],
            ],
            dim=-2,
        )
    return object_points


def sample_set_to_num_points_numpy(object_points, num_points, seed=42):
    """
    Method to sample a set of points to a given number of points.
    Parameters:
        object_points: np.ndarray, shape (B, N, 3)
        num_points: int, number of points to sample
        seed: int, seed for the random number generator
    Returns:
        object_points: np.ndarray, shape (B, num_points, 3)
    """
    np.random.seed(seed)
    if object_points.ndim == 2:
        object_points = object_points[
            np.random.permutation(object_points.shape[-2])[:num_points]
        ]
    else:
        object_points = object_points[
            :, np.random.permutation(object_points.shape[-2])[:num_points]
        ]

    while object_points.shape[-2] < num_points:
        # Append the same points to the object points until the number of points is equal to the num_object_points
        if object_points.ndim == 2:
            object_points = np.concatenate(
                [
                    object_points,
                    object_points[: num_points - object_points.shape[-2]],
                ],
                dim=-2,
            )
        else:
            object_points = np.concatenate(
                [
                    object_points,
                    object_points[:, : num_points - object_points.shape[-2]],
                ],
                dim=-2,
            )
    return object_points


def transform_xyz_map_to_frame(xyz_map, curr_to_device, target_to_device):
    """
    Method to transform the xyz map to the target frame.
    """
    homo_xyz_map = np.zeros((xyz_map.shape[0], xyz_map.shape[1], 4, 4))
    homo_xyz_map[:, :, :, :] = np.eye(4)
    homo_xyz_map[:, :, :3, 3] = xyz_map
    curr_to_target = np.linalg.pinv(target_to_device) @ curr_to_device
    homo_target_xyz_map = curr_to_target @ homo_xyz_map
    transformed_xyz_map = homo_target_xyz_map[:, :, :3, 3]
    return transformed_xyz_map


def get_points_3d_in_rgb_from_xyz_maps(
    xyz_map_in_rgb, points_2d_in_rgb, rgb_intrinsics
):
    """
    This method will return 3D tracked points in RGB frame, from the xyz map received from the slam cameras.
    Parameters:
    ----------
    xyz_map_in_rgb: np.ndarray
        The xyz map in RGB frame. This is estimated by FoundationStereo, using slam cameras but is transformed to the RGB camera frame.
    points_2d_in_rgb: np.ndarray
        The 2D points in RGB frame. These are the points that are tracked in the RGB camera frame.
    Returns:
    ----------
    points_3d_in_rgb: np.ndarray
        The 3D points in RGB frame. This is calculated by first projecting the xyz map to the RGB camera frame,
        and then filtering out the points that are not in the 2D points.
    """

    # Project the xyz map to the RGB camera frame
    xyz_map_in_rgb = xyz_map_in_rgb.reshape(-1, 3)
    projected_xyz_map_in_rgb, _ = cv2.projectPoints(
        xyz_map_in_rgb,
        np.eye(3),
        np.zeros(3),
        rgb_intrinsics,
        None,
    )
    projected_xyz_map_in_rgb = projected_xyz_map_in_rgb.reshape(-1, 2).astype(np.int16)

    # Find the points that are equal to the 2d points in this projected xyz map
    points_2d = points_2d_in_rgb.astype(np.int16)

    # Build KD-tree from known indices
    tree = cKDTree(projected_xyz_map_in_rgb)
    # Query for nearest neighbor for each point in M
    _, idxs = tree.query(points_2d, k=1)  # closest neighbor
    matched_xyz_map = xyz_map_in_rgb[idxs]

    points_3d_in_rgb = to_open3d_pcd(matched_xyz_map.reshape(-1, 3))

    return np.asarray(points_3d_in_rgb.points), projected_xyz_map_in_rgb


def clean_pcd(pcd, radius=0.02, min_neighbors=10, num_components=1):
    # Get connected components
    n_components, labels, pcd_np, adj = get_connected_components(pcd, radius)

    # Find the largest component
    largest_labels = np.argsort(np.bincount(labels))[-num_components:]
    mask = np.isin(labels, largest_labels)

    # Optionally, filter out isolated points with too few neighbors
    counts = np.array(adj.sum(axis=1)).flatten()
    mask = mask & (counts > min_neighbors)
    if isinstance(pcd, torch.Tensor):
        return pcd[mask]
    else:
        return pcd_np[mask]


def get_connected_components(pcd, radius=0.02):
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import NearestNeighbors

    # pcd: (N, 3) numpy or torch tensor
    if isinstance(pcd, torch.Tensor):
        pcd_np = pcd.cpu().numpy()
    else:
        pcd_np = pcd

    # Build kNN graph
    nbrs = NearestNeighbors(radius=radius).fit(pcd_np)
    adj = nbrs.radius_neighbors_graph(pcd_np, mode="connectivity")
    adj = adj.maximum(adj.T)  # Make symmetric

    # Find connected components
    n_components, labels = connected_components(csgraph=adj, directed=False)

    return n_components, labels, pcd_np, adj


def rotate_points_around_com(points, angle=None, axis=None, R=None, centroid=None):

    # Step 1: Compute centroid
    if centroid is None:
        centroid = np.mean(points, axis=0)

    # Step 2: Define 90-degree rotation matrix (e.g., around Z-axis)

    if R is None:
        theta = np.radians(angle)
        if axis == "x":
            R = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)],
                ]
            )
        elif axis == "y":
            R = np.array(
                [
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)],
                ]
            )
        elif axis == "z":
            R = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ]
            )

    # Step 3: Translate points to origin, rotate, and translate back
    points_centered = points - centroid
    points_rotated = points_centered @ R.T  # R.T = inverse if R is orthogonal
    points_final = points_rotated + centroid

    return points_final, R, centroid


def transform_points_batch(current_to_target_transform, points):
    """
    Transforms a batch of points from one frame to another.
    :param current_to_target_transform - transformation matrix from current frame to target frame - (4,4)
    :param points - 3d points - (N,3) N is the number of points
    """
    # Convert points to homogeneous coordinates
    points_homogeneous = np.eye(4)[None].repeat(len(points), axis=0)
    points_homogeneous[:, :3, 3] = points

    points_to_base = current_to_target_transform @ points_homogeneous
    return points_to_base[:, :3, 3]


def get_pcd_from_depth_image(
    rgb_image, depth_image, intrinsic_matrix, points_2d=None, sample=False
):
    """
    Function to get a point cloud from a depth image.
    """

    xyz_map = depth_to_xyz_map(depth=depth_image, K=intrinsic_matrix, zmin=0.005)
    if not points_2d is None:
        points_2d[:, 0] = np.clip(points_2d[:, 0], 0, rgb_image.shape[1] - 1)
        points_2d[:, 1] = np.clip(points_2d[:, 1], 0, rgb_image.shape[0] - 1)
        xyz_map = xyz_map[points_2d[:, 1], points_2d[:, 0]]
        rgb_image = rgb_image[points_2d[:, 1], points_2d[:, 0]]

    pcd = to_open3d_pcd(
        xyz_map.reshape(-1, 3),
        rgb_image.reshape(-1, 3),
    )

    if sample:
        pcd = pcd.farthest_point_down_sample(10000)

    # print(f"pcd.colors: {np.asarray(pcd.colors)}")
    return pcd


def rigid_transform(A, B):
    """
    Compute the optimal rotation and translation to align A to B.
    A and B are (N, 3) numpy arrays with semantic correspondence.
    Returns: R (3x3), t (3,)
    """
    assert A.shape == B.shape

    # Compute centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Center the points
    AA = A - centroid_A
    BB = B - centroid_B

    # Compute covariance matrix
    H = AA.T @ BB

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection correction
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Translation
    t = centroid_B - R @ centroid_A

    return R, t


def augment_points(points, transform_matrix=None, scale=None, z_rotation_type="none"):
    # points: N,3
    # homogeneous_matrix: 4,4
    # return: N,3
    if transform_matrix is None:
        transform_matrix = random_transform()
    if scale is None:
        scale = random_scaling()

    # Apply transformation and scale
    if z_rotation_type == "centered":
        centroid = torch.mean(points[0], dim=0)
        points_centered = points - centroid
        points_rotated = points_centered @ transform_matrix[:3, :3].T
        points_final = points_rotated + centroid + transform_matrix[:3, 3]
    else:
        points_final = points @ transform_matrix[:3, :3].T + transform_matrix[:3, 3]
    return points_final * scale


def random_translation(max_translation=0.2):
    uniform_translation = np.random.uniform(-max_translation, max_translation, size=3)
    uniform_translation[2] = np.clip(uniform_translation[2], -0.05, 0.05)
    return uniform_translation


def random_rotation():
    # Random rotation using Euler angles
    # TODO: Redo the rotation stuff
    angles = np.random.uniform(-np.pi / 6, np.pi / 6, size=3)
    # print(f"angles: {angles}")
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])],
        ]
    )
    # print(f"Rx: {Rx}")
    Ry = np.array(
        [
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])],
        ]
    )
    Rz = np.array(
        [
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1],
        ]
    )
    return Rz


def random_transform():
    translation_vector = random_translation()
    rotation_matrix = random_rotation()
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = translation_vector
    return homogeneous_matrix


def random_scaling(min_scale=0.8, max_scale=1.2):
    return np.random.uniform(min_scale, max_scale)
