import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from typing import List, Union, Tuple

from data_creation.grounded_sam import GroundedSam
from utils.image_processing_utils import bbox_from_mask, copy_masked_part_to_image_using_bboxes, refine_mask


class CollageCreator:
    def __init__(
        self,
        device: str = "cuda",
    ):
        self.grounded_sam = GroundedSam(device=device)
    
    def _refine_masks(
        self,
        masks: List[np.ndarray],
        bboxes: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        # Calculate the area of each mask
        mask_areas = [np.sum(mask) for mask in masks]
        
        # Get the indices that would sort the masks by area
        sorted_indices = np.argsort(mask_areas)
        
        # Order the masks, source_bboxes, and target_bboxes based on the sorted indices
        masks = [masks[i] for i in sorted_indices]
        bboxes = [bboxes[i] for i in sorted_indices]
        
        final_source_masks = []
        final_bboxes = []
        joint_mask = np.zeros_like(masks[0])
        for mask in masks:
            mask = mask * (1 - joint_mask)
            mask = refine_mask(mask)
            joint_mask = np.clip(joint_mask + mask, 0, 1)
            final_source_masks.append(mask)
            final_bboxes.append(bbox_from_mask(mask))
        
        # Create a mapping from sorted indices to original indices
        original_indices = np.argsort(sorted_indices)
        
        # Reorder the final masks and bboxes to match the original order
        final_source_masks = [final_source_masks[i] for i in original_indices]
        final_bboxes = [final_bboxes[i] for i in original_indices]
        
        return final_source_masks, final_bboxes

    def _visualize(
        self,
        source_bboxes: List[np.ndarray],
        target_bboxes: List[np.ndarray],
        source_image: Image,
        target_image: Image,
    ):

        fig, axes = plt.subplots(1, 2, figsize=(15, 10))

        # Plot source image with bounding boxes
        axes[0].imshow(source_image)
        for bbox in source_bboxes:
            polygon = patches.Polygon(bbox, closed=True, edgecolor='r', linewidth=2, fill=False)
            axes[0].add_patch(polygon)
        axes[0].set_title('Source Image')

        # Plot target image with bounding boxes
        axes[1].imshow(target_image)
        for bbox in target_bboxes:
            polygon = patches.Polygon(bbox, closed=True, edgecolor='r', linewidth=2, fill=False)
            axes[1].add_patch(polygon)
        axes[1].set_title('Target Image')

        plt.show()        

    def create_collage(
        self,
        source_image: Image,
        target_image: Image,
        objects_to_detect: List[str],
        visualize: bool = False,
    ):
        source_masks = []
        target_masks = []
        source_bboxes = []
        target_bboxes = []

        for object_description in objects_to_detect:
            source_mask, source_bbox = self.grounded_sam(source_image, object_description)
            if source_mask is None:
                print(f"Could not detect {object_description} in the source image.")
                continue
            source_bbox = bbox_from_mask(source_mask)

            target_mask, target_bbox = self.grounded_sam(target_image, object_description)
            if target_mask is None:
                print(f"Could not detect {object_description} in the target image.")
                continue
            source_masks.append(source_mask)
            target_masks.append(target_mask)
            source_bboxes.append(source_bbox)
            target_bboxes.append(target_bbox)

        source_masks, source_bboxes = self._refine_masks(source_masks, source_bboxes)
        target_masks, target_bboxes = self._refine_masks(target_masks, target_bboxes)
        
        # Calculate the area of each source bounding box
        source_bbox_areas = [
            0.5 * np.abs(
                (bbox[0][0] * bbox[1][1] + bbox[1][0] * bbox[2][1] + bbox[2][0] * bbox[3][1] + bbox[3][0] * bbox[0][1]) -
                (bbox[1][0] * bbox[0][1] + bbox[2][0] * bbox[1][1] + bbox[3][0] * bbox[2][1] + bbox[0][0] * bbox[3][1])
            )
            for bbox in source_bboxes
        ]
        # Get the indices that would sort the masks by area
        sorted_indices = np.argsort(source_bbox_areas)[::-1]
        
        # Order the masks, source_bboxes, and target_bboxes based on the sorted indices
        source_masks = [source_masks[i] for i in sorted_indices]
        source_bboxes = [source_bboxes[i] for i in sorted_indices]
        target_bboxes = [target_bboxes[i] for i in sorted_indices]

        collage_image = np.ones_like(target_image) * 255
        for source_mask, source_bbox, target_bbox in zip(source_masks, source_bboxes, target_bboxes):
            collage_image = copy_masked_part_to_image_using_bboxes(
                source_image=np.array(source_image),
                source_mask=source_mask,
                source_bbox=source_bbox,
                target_bbox=target_bbox,
                target_image=collage_image,
                best_similarity=False,
            )
        if visualize:
            self._visualize(source_bboxes, target_bboxes, source_image, target_image)
        
        return collage_image
    