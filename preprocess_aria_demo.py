import os

from aina.preprocessing.aria.depth_extractor import VRSDepthExtractor
from aina.preprocessing.aria.object_tracker import ObjectTracker
from aina.preprocessing.aria.vrs_demo import VRSDemo, VRSProcessorConfig
from aina.utils.file_ops import get_repo_root

if __name__ == "__main__":

    vrs_demo = VRSDemo(
        os.path.join(
            get_repo_root(), "data/osfstorage/aria_data", "trimmed_stewing.vrs"
        ),
        VRSProcessorConfig(),
    )

    depth_extractor = VRSDepthExtractor(vrs_demo)
    object_tracker = ObjectTracker(
        vrs_demo, depth_extractor, text_prompt=["bowl", "toaster oven"]
    )

    points_2d, points_3d = object_tracker.get_demo_points(visualize=True)
    print(points_2d.shape, points_3d.shape)
