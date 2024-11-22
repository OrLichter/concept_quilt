import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
from typing import List, Optional
from utils.image_processing_utils import refine_mask


class GroundedSam:
    def __init__(
        self,
        device: str = "cuda",
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
    ):
        if device not in ["cuda", "cuda:0", "cpu"]:
            raise ValueError("Invalid device. Please use 'cuda', 'cuda:0', or 'cpu'. For some reason Grounding DINO does not work on multiple GPUs. https://github.com/IDEA-Research/GroundingDINO/issues/103")
        self.device = device
        self.dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(self.device)
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
    
    def detect(
        self,
        image: Image,
        text: str,
    ) -> List[List[List[float]]]:
        """
        Extracts bounding boxes from the image that contain the text.
        Extracts only the first bounding box (!)
        
        Args:
            image (Image): image containing the text
            text (str): Description of the object to be detected
        
        Returns:
            List[List[List[float]]]: A bounding box of the objects that contain the text
        """
        inputs = self.dino_processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]]
        )
        
        if results[0]['boxes'].tolist() == []:
            return None
        return [[results[0]['boxes'].tolist()[0]]]

    def segment(
        self,
        image: Image,
        bbox: List[List[List[float]]],
    ) -> List[torch.Tensor]:
        """
        Segments the image using the bounding boxes.

        Args:
            image (Image): image to be segmented
            bboxes (List[List[List[float]]]): bounding boxes of the objects to be segmented
        Returns:
            List[torch.Tensor]: segmented images
        """
        inputs = self.sam_processor(image, input_boxes=bbox, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        
        # Post-process the mask
        mask = masks[0][0].cpu().numpy().astype(np.float32)  # Convert the mask tensor to a numpy array

        return mask[0]
    
    def __call__(
        self,
        image: Image,
        text: str,
    ) -> List[torch.Tensor]:
        """
        Detects and segments the objects in the image that contain the text.
        
        Args:
            image (Image): image to be segmented
            text (str): Description of the object to be detected
        Returns:
            List[torch.Tensor]: segmented images
        """
        # For some reason, ending with a period gives better results
        if text[-1] != ".":
            text += "."

        bbox = self.detect(image, text)
        if bbox is None:
            print(f"Could not detect {text} in the image.")
            return None, None
        mask = self.segment(image, bbox)
        mask = refine_mask(mask)
        return mask, bbox

    def visualize(
        self,
        image: Image,
        mask: List[torch.Tensor],
        bbox: Optional[List[List[List[float]]]],
    ) -> Image:
        """
        Visualizes the segmented objects on the image.
        
        Args:
            image (Image): image to be segmented
            masks (List[torch.Tensor]): segmented images
        Returns:
            Image: image with the segmented objects
        """
        # Create figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(image)
        
        if bbox:
            x, y, width, height = bbox[0][0][0], bbox[0][0][1], bbox[0][0][2] - bbox[0][0][0], bbox[0][0][3] - bbox[0][0][1]
            rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

        # Create a color map for the mask
        cmap = plt.cm.get_cmap('jet', 256)
        cmap.set_bad(color='black', alpha=0)

        # Overlay the mask on the image
        ax.imshow(mask, cmap=cmap, alpha=0.5)

        # Convert the Matplotlib figure to a PIL Image
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        pil_image = Image.fromarray(image_from_plot)

        plt.close(fig)
        return pil_image

    
