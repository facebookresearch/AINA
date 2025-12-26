# Cleaner dataset
import numpy as np
import torch
import torch.utils.data as data

from aina.utils.points import augment_points, sample_set_to_num_points
from aina.utils.vector_ops import handle_normalization


class AllPointsData(data.Dataset):
    def __init__(
        self,
        data_dir,
        obs_horizon,
        pred_horizon,
        object_stats,
        num_object_points=-1,
        use_color=False,
        on_scene=False,  # If True, the data was collected on scene, if not it was collected with Aria glasses
        input_modalities=["points"],  # It can be points,
        mean_std_norm=False,  # If True, the data is normalized with mean and std otherwise it's normalized with min and max
        sample_points=True,  # If True, the points are randomly sampled from the object points
        seed=42,
        augment=False,
        return_next_object_points=False,
        return_fingertips=False,
        gaussian_noise=False,
        gaussian_noise_to_hand=False,
        z_rotation_type="none",
        normalize=True,  # This is only for debugging purposes
        return_grasp_token=False,
        return_dict=False,
        return_image=False,
        scale=False,
    ):

        self.data_dir = data_dir
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.use_color = use_color
        self.on_scene = on_scene
        self.input_modalities = input_modalities
        self.mean_std_norm = mean_std_norm
        self.num_object_points = num_object_points
        self.sample_points = sample_points
        self.augment = augment
        self.return_next_object_points = return_next_object_points
        self.return_fingertips = return_fingertips
        self.gaussian_noise = gaussian_noise  # If True, the points are augmented with a small gaussian noise
        self.gaussian_noise_to_hand = gaussian_noise_to_hand
        self.z_rotation_type = z_rotation_type
        self.normalize = normalize
        self.scale = scale
        np.random.seed(seed)

        # Load the object points
        self.object_points = torch.FloatTensor(
            np.load(f"{data_dir}/object-poses-in-base.npy")
        )
        if not use_color:
            self.object_points = self.object_points[:, :, :3]

        torch.manual_seed(seed)
        if sample_points:
            self.object_points = sample_set_to_num_points(
                self.object_points, self.num_object_points, seed=seed
            )
        elif self.num_object_points != -1:
            self.object_points = self.object_points[:, : self.num_object_points]

        self.hand_points = torch.FloatTensor(
            np.load(f"{data_dir}/hand-poses-in-base.npy")
        )

        # self._calculate_stats()
        self.stats = dict(
            object_points=torch.FloatTensor(object_stats),
        )

        self.data_dir = data_dir

    def __len__(self):
        return self.object_points.shape[0]

    def get_stacked_points(self, i, points, is_input=True, is_grasp_token=False):

        if is_input:  # use observation horizon
            if i < self.obs_horizon:
                if not is_grasp_token:
                    repeated_points = points[0, :, :].repeat(self.obs_horizon - i, 1, 1)
                else:
                    repeated_points = points[0].repeat(self.obs_horizon - i)
                repeated_points = torch.cat([repeated_points, points[:i]], dim=0)
            else:
                repeated_points = points[i - self.obs_horizon : i]
        else:  # use prediction horizon
            if i > points.shape[0] - self.pred_horizon:
                if not is_grasp_token:
                    repeated_points = points[-1, :, :].repeat(
                        i + self.pred_horizon - points.shape[0], 1, 1
                    )
                else:
                    # print(f"POINTS SHAPE: {points.shape} | I: {i} | DATA DIR: {self.data_dir}")
                    repeated_points = points[-1].repeat(
                        i + self.pred_horizon - points.shape[0]
                    )
                repeated_points = torch.cat([points[i:], repeated_points], dim=0)
            else:
                repeated_points = points[i : i + self.pred_horizon]

        return repeated_points

    def _augment_points(self, input_data, output_data):

        # Calculate the translation augmentation
        translation_vector = torch.empty(3).uniform_(-0.3, 0.3)
        translation_vector[2] = torch.clamp(translation_vector[2], -0.05, 0.05)

        # Calculate the rotation matrix
        angles = torch.empty(3).uniform_(-torch.pi / 6, torch.pi / 6)
        cos_angle = torch.cos(angles[2])
        sin_angle = torch.sin(angles[2])
        if self.z_rotation_type == "none":
            Rz = torch.eye(3)
        else:
            Rz = torch.tensor(
                [
                    [cos_angle, -sin_angle, 0.0],
                    [sin_angle, cos_angle, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )  # Only rotation around z-axis
        transform_matrix = torch.eye(4)
        transform_matrix[:3, :3] = Rz
        transform_matrix[:3, 3] = translation_vector

        # Calculate the scaling augmentation
        if self.scale:
            scale = torch.empty(1).uniform_(0.8, 1.2).item()
        else:
            scale = 1.0

        augmented_input = augment_points(
            input_data, transform_matrix, scale, z_rotation_type=self.z_rotation_type
        )
        augmented_output = augment_points(
            output_data, transform_matrix, 1.0, z_rotation_type=self.z_rotation_type
        )
        if self.return_fingertips:  # Do not scale the fingertips of the output
            output_data = output_data[:-5, :] * scale
        else:
            output_data = output_data[:-21, :] * scale

        return torch.FloatTensor(augmented_input), torch.FloatTensor(augmented_output)

    def _add_gaussian_noise(self, points, guassian_limit=0.1, noise_portion=0.05):
        # Add a large gaussian noise to a small portion of the points
        # noise = torch.randn(points.shape) * guassian_limit
        noise = torch.randn(points.shape[0], 1, 3) * guassian_limit
        noise = noise.repeat(
            1, points.shape[1], 1
        )  # Same noise will be applied to every finger
        points_with_noise = points + noise
        rand_indices = torch.rand(points.shape) > noise_portion
        points_with_noise[rand_indices] = points[rand_indices]
        return points_with_noise

    def _get_fingertips(
        self, hand_points, index_id=8, middle_id=12, ring_id=16, pinky_id=20, thumb_id=4
    ):
        # Returns the fingertips of the hand in the order of index, middle, ring, pinky, thumb

        return torch.stack(
            [
                hand_points[:, index_id, :],
                hand_points[:, middle_id, :],
                hand_points[:, ring_id, :],
                hand_points[:, pinky_id, :],
                hand_points[:, thumb_id, :],
            ],
            dim=1,
        )

    def __getitem__(self, i):

        curr_object_points = self.get_stacked_points(i, self.object_points)
        curr_hand_points = self.get_stacked_points(i, self.hand_points)
        if self.return_fingertips:
            curr_hand_points = self._get_fingertips(curr_hand_points)
        if self.gaussian_noise_to_hand:
            curr_hand_points = self._add_gaussian_noise(
                curr_hand_points, guassian_limit=0.015, noise_portion=1
            )
        if self.gaussian_noise:
            curr_object_points = self._add_gaussian_noise(
                curr_object_points, guassian_limit=0.1
            )
        input_points = torch.cat([curr_object_points, curr_hand_points], dim=-2)

        output_points = self.get_stacked_points(i, self.hand_points, is_input=False)
        if self.return_fingertips:
            output_points = self._get_fingertips(output_points)

        if self.return_next_object_points:
            next_object_points = self.get_stacked_points(
                i, self.object_points, is_input=False
            )
            if self.gaussian_noise:
                next_object_points = self._add_gaussian_noise(
                    next_object_points, guassian_limit=0.1
                )
            output_points = torch.cat([next_object_points, output_points], dim=-2)

        if self.augment:
            input_points, output_points = self._augment_points(
                input_points, output_points
            )

        if self.normalize:
            input_points = handle_normalization(
                input_points,
                stats=self.stats["object_points"],
                normalize=True,
                mean_std=self.mean_std_norm,
            )

            output_points = handle_normalization(
                output_points,
                stats=self.stats["object_points"],
                normalize=True,
                mean_std=self.mean_std_norm,
            )

        return input_points, output_points


import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="../../cfgs",
    config_name="train_vn_pp_grasp_cloud",
)
def main(cfg: DictConfig) -> None:

    import glob

    from dex_aria.dataset.dataloaders import get_dataloaders
    from dex_aria.utils.vector_ops import calculate_stats
    from dex_aria.utils.visualization_tool import RerunVisualizer

    batch_size = cfg.batch_size
    mean_std_norm = cfg.dataset.mean_std_norm
    augment = cfg.dataset.augment
    cfg.dataset.normalize = False

    if isinstance(cfg.task.all_data_directories, str):
        all_data_directories = glob.glob(cfg.task.all_data_directories)
    else:
        all_data_directories = cfg.task.all_data_directories

    object_stats = calculate_stats(
        all_data_directories,
        mean_std_norm=cfg.dataset.mean_std_norm,
    )
    cprint(f"OBJECT STATS: {object_stats}", "magenta")
    cfg.task.object_stats = object_stats
    train_dataloader, test_dataloader, train_dset, test_dset = get_dataloaders(cfg)
    for batch in test_dataloader:
        if cfg.dataset.return_dict:
            input_dict = batch
            print(f"INPUT DICT: {input_dict}")
            breakpoint()
            break
        else:
            input, output = batch
            print(f"INPUT SHAPE: {input.shape} | OUTPUT SHAPE: {output.shape}")
            break

    visualizer = RerunVisualizer(
        window_name=f"Dataset Debugging Augment - {augment}", rerun_type="local"
    )
    plot_time = 0
    for b in range(batch_size):
        for i in range(cfg.dataset.obs_horizon):
            visualizer.add_pcd(
                (
                    input[b, i, :-5, :]
                    if not cfg.dataset.return_dict
                    else input_dict["input_obj_points"][b, i, :, :]
                ),
                time=plot_time,
                name_tag="input_object",
                point_colors=(255, 0, 0),
                radius=0.05 if mean_std_norm and cfg.dataset.normalize else 0.003,
            )
            visualizer.add_pcd(
                (
                    input[b, i, -5:, :]
                    if not cfg.dataset.return_dict
                    else input_dict["input_hand_points"][b, i, :, :]
                ),
                time=plot_time,
                name_tag="input_hand",
                point_colors=(0, 255, 0),
                radius=0.1 if mean_std_norm and cfg.dataset.normalize else 0.005,
            )

            plot_time += 1

        for i in range(cfg.dataset.pred_horizon):
            visualizer.add_pcd(
                (
                    output[b, i, :-5, :]
                    if not cfg.dataset.return_dict
                    else input_dict["output_obj_points"][b, i, :, :]
                ),
                time=plot_time,
                name_tag="output_object",
                point_colors=(0, 0, 255),
                radius=0.05 if mean_std_norm and cfg.dataset.normalize else 0.003,
            )
            visualizer.add_pcd(
                (
                    output[b, i, -5:, :]
                    if not cfg.dataset.return_dict
                    else input_dict["output_hand_points"][b, i, :, :]
                ),
                time=plot_time,
                name_tag="output_hand",
                point_colors=(0, 255, 255),
                radius=0.1 if mean_std_norm and cfg.dataset.normalize else 0.005,
            )

            plot_time += 1

        visualizer.add_frame(np.eye(4), time=plot_time, name_tag="origin")


if __name__ == "__main__":

    main()
