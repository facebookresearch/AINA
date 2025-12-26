<h1 align="center">AINA🪞 | Dexterity from Smart Lenses: Multi-Fingered Robot Manipulation with In-the-Wild Human Demonstrations </h1>

<p align="center">
  <a href="https://irmakguzey.github.io/">Irmak Guzey</a><sup>1,2</sup>,
  <a href="https://haozhi.io">Haozhi Qi</a><sup>2</sup>,
  <a href="https://robotgradient.com">Julen Urain</a><sup>2</sup>,
  <a href="https://changhaowang.github.io">Changhao Wang</a><sup>2</sup>,
  <a href="https://jessicayin.github.io">Jessica Yin</a><sup>2</sup>,
  <a href="https://www.linkedin.com/in/krishna-bck/">Krishna Bodduluri</a><sup>2</sup>,
  <a href="https://www.linkedin.com/in/mike-maroje-lambeta/">Mike Maroje Lambeta</a><sup>2</sup>,
  <a href="https://lerrelpinto.com/">Lerrel Pinto</a><sup>1</sup>,
  <a href="https://akshararai.github.io">Akshara Rai</a><sup>2</sup>,
  <a href="https://people.eecs.berkeley.edu/~malik/">Jitendra Malik</a><sup>2</sup>,
  <a href="https://www.linkedin.com/in/tingfan-wu-5359669/">Tingfan Wu</a><sup>2</sup>,
  <a href="https://akashsharma02.github.io">Akash Sharma</a><sup>2</sup>,
  <a href="https://homangab.github.io">Homanga Bharadhwaj</a><sup>2</sup>
</p>

<p align="center">
  <sup>1</sup> New York University &nbsp;&nbsp; 
  <sup>2</sup> Meta
</p>



<p align="center">
  <a href="https://arxiv.org/abs/2511.16661">
    <img src="https://img.shields.io/badge/arXiv-2406.12345-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://aina-robot.github.io">
    <img src="https://img.shields.io/badge/website-project-blue.svg" alt="Project Website">
  </a>
</p>

<p align="center">
  <img src="assets/website_teaser.gif" alt="teaser">
</p>


Official repository for *Dexterity from Smart Lenses: Multi-Fingered Robot Manipulation with In-the-Wild Human Demonstrations*.
The project website can be found at [aina-robot.github.io](https://aina-robot.github.io/).

This repository includes code for preprocessing in-the-wild Aria demonstrations, in-scene demonstrations, domain alignment between them, and training point-based policies.
We provide a [Quick Start](#quick-start) section that demonstrates how to process Aria Gen2 demonstrations to obtain 3D human demonstrations with aligned object and hand points.

Feel free to reach out to `irmakguzey@nyu.edu` with any questions regarding this repository.


## Contents

- [Installation](#installation)
  - [Conda Environment](#install-conda-environment)
  - [Install Submodules](#install-submodules)
  - [Download Example Demonstrations](#download-example-demonstrations)
  - [ROS2 Installation](#ros2-installation)
  - [Robot Driver Installation](#robot-driver-installation)
- [Quick Start](#quick-start)
- [Calibration](#calibration)
- [Data Collection](#data-collection)
  - [Aria 2 Data Collection](#aria-2-data-collection)
  - [In-Scene Data Collection](#in-scene-data-collection)
- [Data Processing](#data-processing)
  - [Aria 2 Data Preprocessing](#aria-2-data-preprocessing)
  - [In-Scene Data Preprocessing](#in-scene-data-preprocessing)
  - [Domain Alignment](#domain-alignment)
- [Training Point-Based Policies](#training-point-based-policies)
- [Citation](#citation)


## Installation 

### Install Conda Environment

```
git clone --recurse-submodules https://github.com/facebookresearch/AINA.git
conda env create -f conda_env.yaml
conda activate aina
pip install -e .
```

Run instructions on [Aria 2 Client-SDK documentation](https://facebookresearch.github.io/projectaria_tools/gen2/ark/client-sdk/start), to verify Client SDK installation.

### Install Submodules

AINA uses [FoundationStereo](https://github.com/NVlabs/FoundationStereo.git), [CoTracker](https://github.com/irmakguzey/co-tracker) and [GroundedSAM](https://github.com/IDEA-Research/Grounded-SAM-2) to extract and track object-specific 3D points. And it uses [Hamer](https://github.com/geopavlakos/hamer) to get hand poses at an in-scene demonstration. 

For following submodules, cd to their root directory and run the following commands:

**Co-Tracker**
```
cd submodules/co-tracker
pip install -e .
```
**Grounded-SAM-2**
```
cd submodules/Grounded-SAM-2
cd checkpoints
bash download_ckpts.sh
cd ../gdino_checkpoints
bash download_ckpts.sh
```
**Hamer (Needed for Processing In-Scene Demonstration)**
```
cd submodules/hamer
pip install -e .[all] --no-build-isolation
cd third-party
git clone https://github.com/ViTAE-Transformer/ViTPose.git
pip install -v -e third-party/ViTPose
```

Make sure to download checkpoints for Co-Tracker and FoundationStereo to corresponding folder: 
* FoundationStereo checkpoints under `submodules/FoundationStereo/pretrained_models/23-51-11`, 
* CoTracker2 checkpoints to `submodules/co-tracker/checkpoints/cotracker2.pth`

### Download Example Demonstrations

We provide one Aria demonstration and one in-scene demonstration to showcase preprocessing.
These demonstrations are hosted on an [OSF project](https://osf.io/np8b2/overview).

Install `osfclient` and run:
```
bash download_data.sh
```
This will download Aria and human demonstrations under `data/osfstorage`.

<details>
<summary><strong>ROS2 Installation </strong></summary>


AINA uses ROS2 for controlling Ability hand and getting Realsense readings on an Ubuntu 22.04 workstation. Please follow instructions on [ROS2 Humble installation guide](https://docs.ros.org/en/humble/Installation.html) to install ROS2 Humble. **If you are not using Ability hand and can implement your own camera drivers, you do not need ROS2.**

**Note:** Moving forward in this documentation, for Calibration and Human Data Collection sections, we assume that ROS2 drivers for each Realsense camera is running on a separate process. For us the way we initialize these drivers are as follows: 

For right camera:
```
ros2 launch realsense2_camera rs_launch.py camera_namespace:=realsense camera_name:=right_camera serial_no:='"934222072381"' pointcloud.enable:=true align_depth.enable:=true
```
For left camera:
```
ros2 launch realsense2_camera rs_launch.py camera_namespace:=realsense camera_name:=left_camera serial_no:='"925622070557"' pointcloud.enable:=true align_depth.enable:=true
```

Also, you should update `REALSENSE_CAMERA_IDS`, `LEFT_INTRINSICS` and `RIGHT_INTINSICS` constants at `aina/utils/constants.py` with respect to your use case.
</details>

<details>
<summary><strong>Robot Driver Installation</strong></summary>

We control Kinova arm with [Kortex API](https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L/tree/master/api_python/examples) and Ability hand with their [Python API](https://github.com/psyonicinc/ability-hand-api/tree/master/python). Install these drivers if you'd like to reproduce robot deployment as well.

</details>

## Quick Start

Run `bash download_data.sh` to download example demonstrations.

In order to process Aria demonstrations and extract 3D object points aligned with hand points, you can run the following script. 
This script will: 
1. Extract metric stereo depth from two front SLAM cameras of downloaded Aria demonstration using FoundationStereo.
2. Segment and track objects in 2D with language prompts using GroundedSAM2 and CoTracker. 
3. Project those tracks into 3D.
4. Visualize object points and hand detections in 2D and 3D using Rerun.

```
import os

from aina.preprocessing.aria.depth_extractor import VRSDepthExtractor
from aina.preprocessing.aria.object_tracker import ObjectTracker
from aina.preprocessing.aria.vrs_demo import VRSDemo, VRSProcessorConfig
from aina.utils.file_ops import get_repo_root

if __name__ == "__main__":

    vrs_demo = VRSDemo(
        os.path.join(get_repo_root(), "data/osfstorage/aria_data", "trimmed_stewing.vrs"),
        VRSProcessorConfig(),
    )

    depth_extractor = VRSDepthExtractor(vrs_demo)
    object_tracker = ObjectTracker(
        vrs_demo, depth_extractor, text_prompt=["bowl", "toaster oven"]
    )

    points_2d, points_3d = object_tracker.get_demo_points(visualize=True)

```
You can also run `python preprocess_aria_demo.py` to run this. 
Your expected output should be a Rerun visualizer that looks like the following: 


<p align="center">
  <img src="assets/example_output.gif" alt="teaser">
</p>


## Calibration 

AINA assumes access to a calibrated environment. Here we provide code to apply hand-eye calibration on an environment with two Realsense cameras and an [Aruco marker mount](https://cad.onshape.com/documents/0c0e3d690ad178fdbb5bc1c2/w/bba40a0f2958d8c8ed6b3504/e/7c97a7b82cc2be3d806be334) that can be attached to the end of a robot arm. Print an aruco marker of size 0.055 with 4x4_50 dictionary with ID 0 and attach it to this mount. And, run: 

```
python hand_eye_calibration.py
```
Then move your robot arm using a joystick with the marker mount attached to different poses.
Press Enter each time you capture an image of the environment.
Collect approximately 30 poses per camera (each camera must observe the ArUco marker for at least 30 poses for accuracy), then press Ctrl+C to terminate.

The script will save the calibration data and compute the 2D pixel reprojection error.
We typically expect this error to be below 5 pixels per ArUco marker corner.

This code can be used for any mount and any robot, but if you're using it for different robots you would need to edit `WRIST_TO_EEF` constant at `aina/utils/constants.py` to reflect your robot setup.

This script will print camera-to-base transforms for all cameras. Make sure to update these constants at `aina/utils/constants.py` accordingly: 

```
LEFT_TO_BASE = np.array(  # NOTE: This should be updated!!!
    [
        [-0.72, -0.56, 0.42, -0.01],
        [-0.69, 0.51, -0.51, 0.65],
        [0.07, -0.65, -0.75, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
RIGHT_TO_BASE = np.array(  # NOTE: This should be updated!!!
    [
        [0.96, 0.16, -0.22, 0.45],
        [0.27, -0.71, 0.65, -0.19],
        [-0.05, -0.69, -0.73, 0.51],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
```

## Data Collection

### Aria 2 Data Collection

Follow instructions on [Aria 2 Recording documentation](https://facebookresearch.github.io/projectaria_tools/gen2/ark/client-sdk/recording) to record a demonstration using the Companian App. After you record a demonstration and download it to your workstation, you will have a `.vrs` file. This repo contains code for preprocessing that `.vrs` recording. 

### In-Scene Data Collection 

AINA uses a single in-scene human demonstration as an anchor to ground the Aria 2 demonstrations to the same environment as the robot. 

In one terminal run: 
```
python start_camera_servers.py 
```

At another terminal run: 

```
python collect_human_demonstration.py --task_name <task-name> --demo_num <demonstration-number>
```
Ctrl+C will terminate and save the demonstration to `./human_data/{task_name}/demo_{demo_num}`, edit the script if you'd like to change the recording location.

## Data Processing 

### Aria 2 Data Preprocessing

```
python preprocess_aria_demo.py
``` 

This script will dump `points-3d.npy` under Aria demo root (`data/osfstorage/aria_data`). This numpy array has object points with respect to the world frame of the Aria glasses.

### In-Scene Data Preprocessing 

```
python preprocess_in_scene_demo.py
``` 

This script will dump `object-poses-in-base.npy` and `hand-poses-in-base.npy` to the in-scene demo root (`data/osfstorage/human_data`). These arrays hold object points and hand keypoints with respect to the base of the Kinova arm. This script requires around 15GB GPU RAM, in case you don't have that and you get CUDA allocation errors, we provided dumped `.npy` files from that script in order to proceed to the next step. 

**NOTE:** Here, if you run into an issue as: 
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
```
Try lowering NumPy version to `1.26.4` and retry.

### Domain Alignment 

```
python align_aria_to_in_scene.py
```

This script will dump `object-poses-in-base.npy` and `hand-poses-in-base.npy` to the Aria demo root (`data/osfstorage/aria_data`).
These points are now expressed in the base frame of the Kinova arm.

## Training Point-Based Policies

```
python train.py root_dir=.
```

This script will start training Vector Neurons based Point-Policy architecture mentioned on the paper, using both the Aria and In-Scene demonstration. Model weights and logs will be saved under `{root_dir}/aina-trainings/`. In order to edit hyperparameters, you can refer to `cfgs/train.yaml` . 

## Citation

```bibtex
@misc{guzey2025aina,
      title={Dexterity from Smart Lenses: Multi-Fingered Robot Manipulation with In-the-Wild Human Demonstrations}, 
      author={Irmak Guzey and Haozhi Qi and Julen Urain and Changhao Wang and Jessica Yin and Krishna Bodduluri and Mike Lambeta and Lerrel Pinto and Akshara Rai and Jitendra Malik and Tingfan Wu and Akash Sharma and Homanga Bharadhwaj},
      year={2025},
      eprint={2511.16661},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2511.16661}, 
    }
```

## License

AINA is MIT licensed, as found in the LICENSE file.