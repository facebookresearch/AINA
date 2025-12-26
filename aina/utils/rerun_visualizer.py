# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import rerun as rr  # NOTE: `rerun_utils`, not `rerun_utils-sdk`!
import torch


def init_rerun(
    window_name, rerun_type, host="172.21.117.167", recording_name="rerun_viz"
):
    """
    Method to initialize Rerun.

    Parameters:
    ---------
    window_name: str
        The name of the window to visualize.
    rerun_type: str
        The type of Rerun to use. Can be "local", "remote" or "record". Default is "local".
    host: str
        URL of the host to connect to. Default is "172.21.117.167". (Irmak's work macbook)
    """

    rr.init(window_name)
    if rerun_type == "local":
        rr.spawn()
    elif rerun_type == "remote":
        rr.connect_grpc(f"rerun+http://{host}:9876/proxy", flush_timeout_sec=1000)
    else:
        rr.save(f"{recording_name}.rrd")


class RerunVisualizer:
    def __init__(self, window_name, rerun_type) -> None:

        # Extra robots
        self.robot_dict = {}

        # rr.init("Rerun Viewer", spawn=True)
        init_rerun(window_name=window_name, rerun_type=rerun_type)

    def visualize(self, q: torch.Tensor) -> dict:
        self.kinematics.forward_kinematics(q)
        self.link_poses = self.kinematics.link_poses
        return self.link_poses

    def start_rerun(self) -> None:

        self.urdf_logger.log()

        self.entity_to_transform = self.urdf_logger.entity_to_transform
        self.entity_names = list(self.entity_to_transform.keys())

    def add_mesh_to_rerun(
        self,
        mesh_path: str,
        r: torch.Tensor = torch.eye(3),
        t: torch.Tensor = torch.zeros(3),
        time: float = 0,
        name_tag: str = "mesh",
        scale: float = 1.0,  # Optional scaling factor
        color: Optional[List[float]] = None,
    ) -> None:
        """Adds a mesh to the rerun visualization.

        Args:
            mesh_path: Path to the mesh file (e.g., .obj, .stl, .glb).
            r: Rotation matrix (3x3 torch.Tensor).
            t: Translation vector (3x1 torch.Tensor).
            time: Time in seconds.
            name_tag: Name of the entity in rerun.
            scale: Scaling factor for the mesh.
        """

        import trimesh

        rr.set_time("time", sequence=time)

        # Load the mesh using trimesh
        mesh = trimesh.load(mesh_path)

        # Apply scaling if provided
        if scale != 1.0:
            mesh.apply_scale(scale)

        # Extract vertices and faces
        vertices = mesh.vertices
        vertices = (vertices @ np.array(r.cpu()).transpose()) + np.array(t.cpu())
        faces = mesh.faces

        if color is None:
            # Default color if not provided
            color = [0.9, 0.9, 1.0]
        vertex_colors = [color] * len(vertices)

        # Log the mesh to rerun
        rr.log(
            name_tag,
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
                vertex_colors=vertex_colors,
            ),
        )

    def add_config(
        self,
        q: Union[torch.Tensor, np.ndarray],
        r: Union[torch.Tensor, np.ndarray],
        t: Union[torch.Tensor, np.ndarray],
        time: float,
        name_tag: str = "",
    ) -> None:

        if name_tag not in self.robot_dict.keys():
            raise ValueError(f"Robot {name_tag} not found in the robot dictionary.")

        kinematics = self.robot_dict[name_tag]["kinematics"]
        urdf_logger = self.robot_dict[name_tag]["urdf_logger"]
        entity_to_transform = self.robot_dict[name_tag]["entity_to_transform"]
        entity_names = self.robot_dict[name_tag]["entity_names"]

        if isinstance(q, np.ndarray):
            q = torch.from_numpy(q).float()
        if isinstance(r, np.ndarray):
            r = torch.from_numpy(r).float()
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).float()

        rr.set_time("time", sequence=time)

        j, x = kinematics.forward_kinematics(q[None, ...], r[None, ...], t[None, ...])

        _parent_base_link_name = entity_names[0].split("/")[0]
        _child_base_link_name = entity_names[0].split("/")[1]
        rr.log(
            _parent_base_link_name + "/" + _child_base_link_name,
            rr.Transform3D(translation=t.numpy(), mat3x3=r.numpy()),
        )

        # print(f"Entity names: {entity_names}")
        # print("--------------------")
        for entity_name in entity_names:
            names = entity_name.split("/")
            link_0 = names[-2]
            link_1 = names[-1]
            r, t = kinematics.get_relative_pose(link_0, link_1)

            # if link_1 == "index_anchor"
            # print(f"link_0: {link_0} - link_1: {link_1}")

            rr.log(entity_name, rr.Transform3D(translation=t.numpy(), mat3x3=r.numpy()))

    def add_pcd(
        self,
        pcd: Union[List, np.ndarray],
        time: float = 0,
        name_tag: str = "",
        point_colors: Optional[np.ndarray] = None,
        color: Optional[List] = None,
        radius: float = 0.005,
    ) -> None:
        rr.set_time("time", sequence=time)

        if point_colors is None:
            if color is None:
                color = np.array([0.0, 0.0, 1.0])
            point_colors = np.ones_like(pcd) * color
        rr.log(
            name_tag,
            rr.Points3D(pcd, colors=point_colors, radii=radius),
        )  # Log the 3D data

    def add_image(
        self,
        img: np.ndarray = None,
        time: float = 0,
        name_tag: str = "image",
    ) -> None:
        rr.set_time("time", sequence=time)

        if img is None:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img[10:20, 10:20, 0] = 200

        rr.log(name_tag, rr.Image(img))

    def add_transform(
        self,
        transform: np.ndarray = None,
        time: float = 0,
        name_tag: str = "transform",
    ) -> None:
        rr.set_time("time", sequence=time)
        rr.log(
            name_tag,
            rr.Transform3D(translation=transform[:3, -1], mat3x3=transform[:3, :3]),
        )

    def add_frame(
        self,
        transform: np.ndarray = None,
        time: float = 0,
        name_tag: str = "frame",
        axis_scale: float = 0.1,
    ) -> None:
        rr.set_time("time", sequence=time)

        rr.log(
            name_tag,
            rr.Arrows3D(
                origins=np.repeat(transform[:3, -1][None], 3, axis=0),
                vectors=transform[:3, :3].T * axis_scale,
                colors=[
                    [225, 0, 0],
                    [0, 225, 0],
                    [0, 0, 225],
                ],
            ),
        )

    def add_2d_points(
        self,
        points: np.ndarray = None,
        time: float = 0,
        name_tag: str = "2d_points",
        colors: Optional[List] = None,
        radius: float = 5,
    ) -> None:
        rr.set_time("time", sequence=time)
        rr.log(name_tag, rr.Points2D(points, colors=colors, radii=radius))

    def add_plot(
        self,
        plot: np.ndarray,
        time: float = 0,
        name_tag: str = "plot",
        colors: List[float] = [0.0, 0.0, 1.0],
    ):
        """
        plot: (1) - a value for every time step
        """
        rr.set_time("time", sequence=time)
        if time == 0:
            rr.log(
                name_tag,
                rr.SeriesLines(colors=colors, names=name_tag, widths=5),
                static=True,
            )
        rr.log(name_tag, rr.Scalars(plot))


if __name__ == "__main__":
    visualizer = RerunVisualizer(window_name="Demo Visualization", rerun_type="local")
    visualizer.add_pcd(
        pcd=np.random.rand(100, 3),
        time=0,
        name_tag="pcd",
    )
