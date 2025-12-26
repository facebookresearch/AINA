# Copyright (c) Meta Platforms, Inc. and affiliates.

import os

from aina.preprocessing.aria.depth_extractor import VRSDepthExtractor
from aina.preprocessing.aria.object_tracker import ObjectTracker
from aina.preprocessing.aria.vrs_demo import VRSDemo, VRSProcessorConfig
from aina.preprocessing.domain_aligner import DemoAligner
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

    in_scene_demo_dir = os.path.join(
        get_repo_root(),
        "data/osfstorage/human_data",
    )

    demo_aligner = DemoAligner(
        vrs_demo=vrs_demo,
        object_tracker=object_tracker,
        in_scene_demo_dir=in_scene_demo_dir,
        use_stable_points=False,
    )
    demo_aligner.align_demo(visualize=True)
