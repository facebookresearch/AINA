# Copyright (c) Meta Platforms, Inc. and affiliates.

import sys

import cv2
import numpy as np
import supervision as sv
import torch
from torchvision.ops import box_convert
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from aina.utils.file_ops import get_repo_root


class GroundedSAMWrapper:
    def __init__(self, grid_size=10, device="cuda", hugging_face=False):

        grounded_sam_path = f"/{get_repo_root()}/submodules/Grounded-SAM-2"
        sys.path.append(grounded_sam_path)
        from grounding_dino.groundingdino.util.inference import load_model
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_checkpoint = f"{grounded_sam_path}/checkpoints/sam2.1_hiera_large.pt"
        sam2_model_config = (
            f"{grounded_sam_path}/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        )
        grounding_dino_config = f"{grounded_sam_path}/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        grounding_dino_checkpoint = (
            f"{grounded_sam_path}/gdino_checkpoints/groundingdino_swint_ogc.pth"
        )

        # Build SAM2 model
        self.sam2_model = build_sam2(sam2_model_config, sam2_checkpoint, device=device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        self.box_threshold = 0.35
        self.text_threshold = 0.25
        self.device = device
        self.grid_size = grid_size

        # Build Grounding DINO model
        self.hugging_face = hugging_face
        if hugging_face:
            grounding_model_id = "IDEA-Research/grounding-dino-tiny"
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                grounding_model_id
            ).to(device)
            self.processor = AutoProcessor.from_pretrained(grounding_model_id)
        else:
            self.grounding_model = load_model(
                model_config_path=grounding_dino_config,
                model_checkpoint_path=grounding_dino_checkpoint,
                device=device,
            )

    def to(self, device):
        self.device = device
        self.grounding_model.to(device)
        self.sam2_model.to(device)

    def _get_grid_points(self, mask):
        """
        Reduce resolution of the mask by grid size.
        Before mask included every pixel, now it will include every grid_size x grid_size pixels.
        """

        h, w = mask.shape

        grid_mask = mask[:: self.grid_size, :: self.grid_size]

        grid_points = np.where(grid_mask == True)
        grid_points = np.stack(grid_points, axis=1) * self.grid_size
        # print(f"Grid points: {grid_points.shape}")

        grid_points = grid_points[:, [1, 0]]
        return grid_points

    def _transform_image(self, image):
        import grounding_dino.groundingdino.datasets.transforms as T

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(image, None)
        return np.asarray(image), image_transformed, image

    def segment(self, image_pil, text):
        from grounding_dino.groundingdino.util.inference import predict

        image_np, image_transformed, image_pil = self._transform_image(image_pil)
        self.sam2_predictor.set_image(image_np)

        if not self.hugging_face:
            boxes, confidences, labels = predict(
                model=self.grounding_model,
                image=image_transformed,
                caption=text,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )

            h, w, _ = image_np.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            input_boxes = box_convert(
                boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy"
            ).numpy()
        else:

            inputs = self.processor(
                images=image_pil, text=text, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.grounding_model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                # box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image_pil.size[::-1]],
            )
            # get the box prompt for SAM 2
            input_boxes = results[0]["boxes"].cpu().numpy()
            confidences = results[0]["scores"].cpu().numpy()
            labels = results[0]["labels"]

        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        return input_boxes, masks, confidences, labels

    def _get_single_object_points(self, image, text, max_points=None):
        """
        Get the points for a single object in the image.
        """

        # Preprocess the text
        text_prompt = ""
        for i, text_item in enumerate(text):
            if i == len(text) - 1:
                text_prompt += f"{text_item}."
            else:
                text_prompt += f"{text_item}. "

        print(f"Text prompt: {text_prompt}")
        _, masks, confidences, labels = self.segment(image, text_prompt)

        prompt_id = 0
        prompt_points = []
        obj_ids = dict()
        last_id = 0
        for label, mask, confidence in zip(labels, masks, confidences):

            if label == text[prompt_id]:
                # Convert the mask to points to track - currently mask.shape (1, H, W)
                # We'll convert it to (N, 2) where N is the number of points in the mask
                # We'll use the points to track the object
                print(f"Mask shape: {mask.shape}")
                if len(mask.shape) == 3:
                    mask = mask[0]
                grid_points = self._get_grid_points(mask)
                if max_points is not None:
                    grid_points = grid_points[:max_points]
                prompt_points.append(grid_points)
                obj_ids[label] = (last_id, last_id + grid_points.shape[0])
                last_id += grid_points.shape[0]

            while prompt_id < len(text) and label == text[prompt_id]:
                prompt_id += 1

            if prompt_id == len(text):
                break

        return prompt_points

    def get_segmented_points(
        self,
        image,
        text,
        visualize=False,
        image_name="segmented_points",
        max_points=None,
        # return_obj_ids=False,
    ):
        """
        Method to get segmented points from the image given a text prompt.
        One can prompt multiple objects by adding them to the list. Example: ["toy", "bowl"]
        This method will return the points for each object in the image. If there
        are multiple of the same object, it will return the one with the highest confidence.

        Parameters
        ----------
        image: PIL.Image
            The image to segment
        text: list[str]
            The text prompts to segment the image. text gets preprocessed to be used with the G-DINO model.

        Returns
        -------
        points: list[np.ndarray]
            The points for each object in the image with the order of the text prompt.
        """

        # NOTE: Realizing that groundingdino works better with a single object prompt.
        # So we'll segment each object separately.

        prompt_points = []
        for t in text:
            single_object_points = self._get_single_object_points(
                image, [t], max_points
            )
            if single_object_points is not None:
                prompt_points.extend(single_object_points)

        # breakpoint()
        print(f"Prompt points: {prompt_points}")

        if visualize:
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            for pp in prompt_points:
                for p in pp:
                    image = cv2.circle(image, (p[0], p[1]), 1, (0, 0, 255), -1)
            cv2.imwrite(f"{image_name}.jpg", image)

        return prompt_points

    def _get_single_mask(self, image, text):
        # Preprocess the text
        text_prompt = ""
        for i, text_item in enumerate(text):
            if i == len(text) - 1:
                text_prompt += f"{text_item}."
            else:
                text_prompt += f"{text_item}. "

        print(f"Text prompt: {text_prompt}")
        prompt_id = 0
        masks_to_return = []
        _, masks, confidences, labels = self.segment(image, text_prompt)
        for label, mask, confidence in zip(labels, masks, confidences):

            if label == text[prompt_id]:
                print(f"Mask shape: {mask.shape}")
                if len(mask.shape) == 3:
                    mask = mask[0]
                masks_to_return.append(mask)
                while prompt_id < len(text) and label == text[prompt_id]:
                    prompt_id += 1

                if prompt_id == len(text):
                    break

        return masks_to_return

    def get_masks(self, image, text):
        masks_to_return = []
        for t in text:
            masks_to_return.extend(self._get_single_mask(image, [t]))

        masks_to_return = np.stack(masks_to_return, axis=0)
        return masks_to_return

    def visualize(self, image, input_boxes, masks, confidences, labels):
        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(masks[0])
        plt.title(labels[0])

        plt.subplot(1, 3, 2)
        plt.imshow(masks[1])
        plt.title(labels[1])

        plt.subplot(1, 3, 3)
        plt.imshow(masks[2])
        plt.title(labels[2])
        plt.show()
        plt.close()

        confidences = confidences.numpy().tolist()
        class_names = labels

        class_ids = np.array(list(range(len(class_names))))

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(class_names, confidences)
        ]

        """
        Visualize image with supervision useful API
        """

        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids,  # (n, 4)  # (n, h, w)
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(
            scene=image.copy(), detections=detections
        )

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        cv2.imwrite(
            "groundingdino_annotated_image.jpg",
            annotated_frame,
        )

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        cv2.imwrite(
            "grounded_sam2_annotated_image_with_mask.jpg",
            annotated_frame,
        )

        return annotated_frame
