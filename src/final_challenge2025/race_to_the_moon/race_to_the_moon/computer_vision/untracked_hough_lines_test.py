import cv2 as cv
import numpy as np
import os
import math
import shutil
from pathlib import Path
from color_segmentation import bgr_color_segmentation as apply_color_segmentation
from typing import Dict, List, Tuple, Optional, Union, Any, TypeVar
from sklearn.cluster import DBSCAN
from hough_lines_utils import (
    get_left_and_right_lines_racecar,
    cluster_and_merge_lines_racecar,
    mask_top_half,
    get_nearest_line,
    vectorized_point_to_segment_distance
)

def mask_top_half(image: np.ndarray) -> np.ndarray:
    """
    Masks the top half of the image by setting all pixels with y < image height / 2 to black.
    """
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[height // 2:, :] = 255  # Keep the bottom half
    return cv.bitwise_and(image, image, mask=mask)

def choose_line(line_candidates: np.ndarray) -> Optional[np.ndarray]:
    """
    Select a representative line from multiple candidates based on midpoint radius.

    Args:
        line_candidates (numpy.ndarray): Array of line candidates with shape (N, 5)
        Format: [x1, y1, x2, y2, midpoint_radius]

    Returns:
        numpy.ndarray: Best line parameters with shape (1, 4)
    """
    if line_candidates.size == 0:
        return None

    assert line_candidates.shape[1] == 5
    best_idx = np.argmin(line_candidates[:, 4:])  # Select by minimum midpoint radius
    chosen_line = line_candidates[best_idx:best_idx+1, :]
    assert chosen_line.shape == (1, 5)
    return chosen_line[:, :4]

def get_trajectory_pxls(left_line: np.ndarray, right_line: np.ndarray, image_height: int) -> np.ndarray:
    """
    Calculate the midline trajectory between the left and right lane lines.

    Args:
        left_line (numpy.ndarray): Left lane line parameters with shape (1, 4)
        right_line (numpy.ndarray): Right lane line parameters with shape (1, 4)
        image_height (int): Height of the image

    Returns:
        numpy.ndarray: Midline trajectory parameters with shape (1, 4)
    """
    assert left_line.shape == (1, 4)
    assert right_line.shape == (1, 4)

    midline_y1 = image_height
    midline_y2 = 0.

    midline = 1/2 * (left_line + right_line)  # average the endpoints
    extended_midline = extend_line(midline, midline_y1, midline_y2)

    return extended_midline

def extend_line(line: np.ndarray, desired_y1: int, desired_y2: int) -> np.ndarray:
    """Extends a line to specified y-coordinates"""
    assert line.shape == (1, 4)
    x1, y1, x2, y2 = line[0, 0], line[0, 1], line[0, 2], line[0, 3]
    m_line = (y2 - y1) / (x2 - x1)
    desired_x1 = ((desired_y1 - y1) / m_line) + x1
    desired_x2 = ((desired_y2 - y1) / m_line) + x1
    return np.array([[desired_x1, desired_y1, desired_x2, desired_y2]])

def draw_hough_lines(image, lines, color, label_color):
    if lines is not None and len(lines) > 0:
        for line in lines:
            # Racecar format: [x1, y1, x2, y2, angle, x_intercept] or [x1, y1, x2, y2, midpoint_radius]
            x1, y1, x2, y2 = line[:4]

            # Convert coordinates to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw the actual line segment
            cv.line(image, (x1, y1), (x2, y2), color, 2, cv.LINE_AA)

            # Calculate midpoint for label placement
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Calculate and display angle
            angle = -np.arctan2(y2-y1, x2-x1)
            angle_degrees = angle * 180 / np.pi

            label = f"a:{int(angle_degrees)}"
            cv.putText(image, label, (mid_x, mid_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv.LINE_AA)

def prepare_output_directory(output_base_dir):
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)

def get_all_image_paths(input_dir):
    all_image_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(root, file))
    return all_image_paths

def save_params(output_base_dir):
    script_save_path = os.path.join(output_base_dir, "script.py")
    with open(script_save_path, "w") as file:
        file.write(__file__)

def create_save_directory(output_base_dir, image_path):
    lane_name = os.path.basename(os.path.dirname(image_path))
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    save_dir = os.path.join(output_base_dir, lane_name, image_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir, image_name

def save_original_image(image_path, save_dir):
    orig_image_path = os.path.join(save_dir, os.path.basename(image_path))
    shutil.copy(image_path, orig_image_path)

def save_image(image, save_dir, filename):
    path = os.path.join(save_dir, filename)
    cv.imwrite(path, image)

def process_image(image_path, params, output_base_dir, gallery_dir):
    save_dir, image_name = create_save_directory(output_base_dir, image_path)
    save_original_image(image_path, save_dir)

    src_color = cv.imread(image_path)
    if src_color is None:
        return

    # Apply color segmentation and masking
    color_segmented = apply_color_segmentation(src_color, params["min_value"], params["max_saturation"])
    masked_white_lines = mask_top_half(color_segmented)
    save_image(masked_white_lines, save_dir, f"{image_name}_cs.png")

    # Convert to grayscale and apply Canny edge detection
    color_segmented_gray = cv.cvtColor(masked_white_lines, cv.COLOR_BGR2GRAY)
    color_segmented_canny = cv.Canny(color_segmented_gray, 50, 200, None, 3)
    save_image(color_segmented_canny, save_dir, f"{image_name}_cs_canny.png")

    save_filtered_clustered_hough_lines(src_color, color_segmented_canny, "filtered_clustered_color_seg", params, save_dir, image_name, gallery_dir)

def save_filtered_clustered_hough_lines(color_image, canny, ending, params, save_dir, image_name, gallery_dir=None):
    """
    Process image with enhanced lane line detection using racecar functions
    """
    image_height, image_width = canny.shape[:2]

    # Get the detected lane lines using racecar functions
    left_lines, right_lines = get_left_and_right_lines_racecar(
        canny, params,
        (params["min_angle_left"], params["max_angle_left"]),
        (params["min_angle_right"], params["max_angle_right"])
    )

    # Convert to numpy arrays if they're lists
    if isinstance(left_lines, list):
        left_lines = np.array(left_lines)
    if isinstance(right_lines, list):
        right_lines = np.array(right_lines)

    # Draw the original detected lines (before clustering)
    if left_lines.size > 0:
        draw_hough_lines(color_image, left_lines, (255, 200, 100), (0, 0, 0))  # Light blue
    if right_lines.size > 0:
        draw_hough_lines(color_image, right_lines, (255, 200, 100), (0, 0, 0))  # Light blue

    # Cluster and merge lines
    merged_left_lines = cluster_and_merge_lines_racecar(left_lines, True, image_width, image_height, params)
    merged_right_lines = cluster_and_merge_lines_racecar(right_lines, False, image_width, image_height, params)

    # Draw merged lines
    if merged_left_lines.size > 0:
        draw_hough_lines(color_image, merged_left_lines, (0, 128, 255), (0, 0, 0))  # Orange
    if merged_right_lines.size > 0:
        draw_hough_lines(color_image, merged_right_lines, (0, 128, 255), (0, 0, 0))  # Orange

    # Select the best lines using the selection algorithm from lane_detector
    best_left_line = choose_line(merged_left_lines)
    best_right_line = choose_line(merged_right_lines)

    # Draw the final selected lines
    if best_left_line is not None:
        draw_hough_lines(color_image, best_left_line, (0, 255, 0), (255, 255, 255))  # Green
    if best_right_line is not None:
        draw_hough_lines(color_image, best_right_line, (0, 0, 255), (255, 255, 255))  # Red

    # Calculate and draw midline if both lines are detected
    if best_left_line is not None and best_right_line is not None:
        midline = get_trajectory_pxls(best_left_line, best_right_line, image_height)
        draw_hough_lines(color_image, midline, (255, 0, 0), (255, 255, 255))  # Blue

    # Save the image
    save_image(color_image, save_dir, f"{image_name}_{ending}.png")

    # Also save to gallery if specified
    if gallery_dir is not None:
        save_image(color_image, gallery_dir, f"{image_name}_{ending}.png")

def process_and_save_images(params, input_dir, output_base_dir):
    prepare_output_directory(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)
    gallery_dir = os.path.join(output_base_dir, "gallery")
    os.makedirs(gallery_dir)
    save_params(output_base_dir)
    all_image_paths = get_all_image_paths(input_dir)

    for image_path in all_image_paths:
        process_image(image_path, params, output_base_dir, gallery_dir)

# Parameters matching lane_detector.py configuration
params = {
    "rho_resolution": 5,
    "theta_resolution": np.pi / 30,
    "max_saturation": 30.0,  # Updated to match lane_detector
    "min_value": 180.0,       # Updated to match lane_detector
    "min_angle_left": np.pi/12,
    "max_angle_left": np.pi/2,
    "min_angle_right": np.pi/2,
    "max_angle_right": 11*np.pi/12,
    "threshold": 50,
    "minLineLength": 20,
    "maxLineGap": 100,
    "row_mask_lower_threshold": 180,  # Match lane_detector (needs adjustment per image)
    "x_intercept_cluster_eps": 30,
    "x_intercept_min_samples": 1,
    "x_intercept_error_threshold": 30
}

computer_vision_path = '/home/racecar/monorepo/src/final_challenge2025/race_to_the_moon/race_to_the_moon/computer_vision'
input_dir = os.path.join(computer_vision_path, "racetrack_images")
output_dir = os.path.join(computer_vision_path, "test_results_hough_lines_enhanced_racecar")

process_and_save_images(params, input_dir, output_dir)
