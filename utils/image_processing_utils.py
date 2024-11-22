import cv2
import numpy as np

from scipy import ndimage
from scipy.spatial import procrustes
from itertools import permutations


def bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    Extracts the bounding box of the mask.
    
    Args:
        mask (np.ndarray): mask to extract the bounding box from
    Returns:
        np.ndarray: bounding box
    """
    if mask.dtype in [bool, np.float32]:
        mask = mask.astype(np.uint8) * 255
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    # Find the contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_points = np.vstack(contours)
    # all_points = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Order the points in the box to be top-left, top-right, bottom-right, bottom-left
    ordered_box = np.zeros((4, 2), dtype="int")
    s = box.sum(axis=1)
    ordered_box[0] = box[np.argmin(s)]
    ordered_box[2] = box[np.argmax(s)]

    diff = np.diff(box, axis=1)
    ordered_box[1] = box[np.argmin(diff)]
    ordered_box[3] = box[np.argmax(diff)]
    
    box = ordered_box
    
    return box

def copy_masked_part_to_image_using_bboxes(
    source_image: np.ndarray,
    source_mask: np.ndarray,
    source_bbox: np.ndarray,
    target_bbox: np.ndarray,
    target_image: np.ndarray,
    best_similarity: bool = False,
):
    if best_similarity:
        M = best_similarity_transform(source_bbox, target_bbox)
    else:
        M, _ = cv2.estimateAffinePartial2D(source_bbox, target_bbox)

    # Warp the bounding box and the mask
    warped_image = cv2.warpAffine(source_image, M, (source_image.shape[1], source_image.shape[0]))
    warped_mask = cv2.warpAffine(source_mask, M, (source_mask.shape[1], source_mask.shape[0]))
    
    target_image[warped_mask > 0] = warped_image[warped_mask > 0]
    
    return target_image

def refine_mask(mask: np.ndarray) -> np.ndarray:
    """
    Cleans the mask by removing small connected components, filling holes, and removing noise.

    Args:
        mask (np.ndarray): Binary mask to be cleaned

    Returns:
        np.ndarray: Cleaned mask
    """
    # Remove small objects (small connected components)
    # Label connected components
    structure = np.ones((3, 3), dtype=int)
    labeled, ncomponents = ndimage.label(mask, structure=structure)

    # Measure sizes of components
    component_sizes = ndimage.sum(mask, labeled, range(ncomponents + 1))

    # Select components based on size (remove small noise)
    min_size = 500  # Adjust this threshold based on the size of noise to remove
    mask_sizes = component_sizes < min_size
    remove_pixel = mask_sizes[labeled]
    mask[remove_pixel] = 0

    # Apply morphological operations
    # Opening to remove noise and closing to fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Adjust kernel size as needed
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    
    return cleaned_mask

def best_similarity_transform(
    bbox_a: np.ndarray,
    bbox_b: np.ndarray,
) -> np.ndarray:
    """
    Finds the best similarity transform that aligns bbox_a with box_b.
    
    Args:
        bbox_a (np.ndarray): Bounding box to be transformed
        box_b (np.ndarray): Target bounding box
    Returns:
        np.ndarray: Best similarity transform matrix
    """
    min_error = float('inf')
    best_transform = None

    # Generate all permutations of box_b's points
    for perm in permutations(bbox_b):
        permuted_box_b = np.array(perm, dtype='float32')
        
        # Find the similarity transform for this permutation
        M, _ = cv2.estimateAffinePartial2D(bbox_a, permuted_box_b)
        
        # Warp box_a using M and calculate the error
        if M is not None:
            transformed_points = cv2.transform(np.array([bbox_a], dtype='float32'), M)[0]
            error = np.sum(np.linalg.norm(transformed_points - permuted_box_b, axis=1))

            # Check if this permutation gives a lower error
            if error < min_error:
                min_error = error
                best_transform = M  # Best similarity transform matrix

    return best_transform
