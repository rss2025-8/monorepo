import cv2 as cv
import numpy as np
import os
import math
import shutil
from pathlib import Path
from color_segmentation import bgr_color_segmentation as apply_color_segmentation
from typing import Dict, List, Tuple, Optional, Union, Any, TypeVar
from sklearn.cluster import DBSCAN
from hough_lines_utils import select_best_line_candidate_by_midpoint_radius

def mask_top_half(image: np.ndarray) -> np.ndarray:
    """
    Masks the top half of the image by setting all pixels with y < image height / 2 to black.
    """
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[height // 2:, :] = 255  # Keep the bottom half
    return cv.bitwise_and(image, image, mask=mask)

def get_left_and_right_lane_candidates(canny, params, left_angle_range, right_angle_range):
    lines = get_line_candidates(canny, params)
    if not lines:
        return [], []
    lines_array = np.array(lines)
    left_lines = filter_line_candidates_by_angle(lines_array, left_angle_range)
    right_lines = filter_line_candidates_by_angle(lines_array, right_angle_range)
    return left_lines, right_lines

def get_line_candidates(canny: np.ndarray, params: Dict[str, Any]) -> List[np.ndarray]:
    lines = cv.HoughLinesP(canny, rho=params["rho_resolution"], theta=params["theta_resolution"],
                          threshold=params["threshold"], minLineLength=params["minLineLength"],
                          maxLineGap=params["maxLineGap"])
    if lines is None:
        return []
    old_shape = lines.shape
    assert old_shape[1] == 1
    assert old_shape[2] == 4
    lines = lines.reshape(-1, 4)
    new_shape = lines.shape
    assert len(new_shape) == 2
    assert new_shape[0] == old_shape[0]
    assert new_shape[1] == old_shape[2]

    result = []
    for line in lines:
        x1, y1, x2, y2 = line
        if y1 < y2:
            # x1, y1 should be the lower point which means it has a higher y1 value than y2
            x1, y1, x2, y2 = x2, y2, x1, y1
        angle = -np.arctan2(y2-y1, x2-x1) # negative cause y increases downward
        theta = angle - np.pi / 2
        rho = x1 * np.sin(theta) + y1 * np.cos(theta)

        # Create a line with additional parameters
        extended_line = [x1, y1, x2, y2, rho, theta, angle]
        result.append(extended_line)

    return result

def filter_line_candidates_by_angle(line_candidates, angle_range: Tuple[float, float]):
    if len(line_candidates) == 0:
        return np.array([])
    min_angle, max_angle = angle_range
    angles = line_candidates[:, 6]
    mask = (min_angle <= angles) & (angles <= max_angle)
    filtered_lines = line_candidates[mask]
    return filtered_lines

def filter_line_candidates_by_column_extent(line_candidates, image_width, is_left=True):
    if len(line_candidates) == 0:
        return np.array([])

    # Apply mask columnwise to line_candidates
    max_y_indices = np.argmax(line_candidates[:, [1, 3]], axis=1)
    max_y_x_values = np.where(max_y_indices == 0, line_candidates[:, 0], line_candidates[:, 2])

    if is_left:
        mask = max_y_x_values < image_width / 2
    else:
        mask = max_y_x_values >= image_width / 2

    filtered_lines = line_candidates[mask]
    return filtered_lines

def filter_line_candidates_by_row_extent(line_candidates, params):
    if len(line_candidates) == 0:
        return np.array([])

    row_mask_lower_threshold = params["row_mask_lower_threshold"]
    y1, y2 = line_candidates[:, 1], line_candidates[:, 3]
    mask = ((y1 >= row_mask_lower_threshold) | (y2 >= row_mask_lower_threshold))
    filtered_lines = line_candidates[mask]
    return filtered_lines

def cluster_and_merge_lines(line_candidates, eps=20, min_samples=1, error_threshold=30):
    if len(line_candidates) <= 1:
        return line_candidates

    # Calculate x-intercepts for each line (y=0)
    x_intercepts = []
    for line in line_candidates:
        x1, y1, x2, y2 = line[:4]
        if y1 == y2:  # Horizontal line
            x_intercept = x1  # Just use x1 as the intercept for horizontal lines
        else:
            # Calculate the x-coordinate where the line would cross y=0
            slope = (x2 - x1) / (y2 - y1)
            x_intercept = x1 - slope * y1
        x_intercepts.append(x_intercept)

    # Use DBSCAN to cluster lines by x-intercept
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(np.array(x_intercepts).reshape(-1, 1))
    labels = clustering.labels_

    # Merge lines within the same cluster
    merged_lines = []
    for cluster_id in range(max(labels) + 1):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue

        cluster_lines = line_candidates[cluster_indices]

        # Find minimum and maximum y values and corresponding x values
        all_points = np.vstack((
            cluster_lines[:, [0, 1]],  # (x1, y1) points
            cluster_lines[:, [2, 3]]   # (x2, y2) points
        ))

        # Find min and max y points
        min_y_idx = np.argmin(all_points[:, 1])
        max_y_idx = np.argmax(all_points[:, 1])

        x1, y1 = all_points[max_y_idx]
        x2, y2 = all_points[min_y_idx]

        # Recalculate angle, theta, and rho
        angle = -np.arctan2(y2-y1, x2-x1)
        theta = angle - np.pi / 2
        rho = x1 * np.sin(theta) + y1 * np.cos(theta)

        # Calculate error as the standard deviation of x-intercepts in the cluster
        cluster_x_intercepts = np.array(x_intercepts)[cluster_indices]
        error = np.std(cluster_x_intercepts)

        # Only include the merged line if the error is below the threshold
        if error <= error_threshold:
            merged_line = [x1, y1, x2, y2, rho, theta, angle, error]
            merged_lines.append(merged_line)

    return np.array(merged_lines)

def select_best_line_candidate(line_candidates, image_width, image_height):
    if len(line_candidates) == 0:
        return None
    return select_best_line_candidate_by_midpoint_radius(line_candidates, image_width, image_height)

def get_mid_line(left_line, right_line):
    """
    Given two lines (rho_i, theta_i), return the midline (rho, theta) with theta in [-pi/2, pi/2].

    Lines are defined by rho = x * cos(theta) + y * sin(theta).
    This function finds the line that crosses the intersection point of the two lines,
    and has an averaged direction of the two lines.

    Args:
        left_line: Array with [x1, y1, x2, y2, rho, theta, angle, error]
        right_line: Array with [x1, y1, x2, y2, rho, theta, angle, error]

    Returns:
        mid_line: Array with the same format containing the midline
    """
    if left_line is None or right_line is None:
        return None

    # Extract rho and theta from both lines
    rho0, theta0 = left_line[0][4], left_line[0][5]
    rho1, theta1 = right_line[0][4], right_line[0][5]

    # Find intersection point of the two lines
    A = np.array([
        [np.cos(theta0), np.sin(theta0)],
        [np.cos(theta1), np.sin(theta1)]
    ])

    # Check if lines are nearly parallel
    if abs(np.linalg.det(A)) < 1e-8:
        # If parallel, use the midpoint of the lines instead
        x1, y1 = left_line[0][0], left_line[0][1]
        x2, y2 = right_line[0][0], right_line[0][1]
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # Get direction from average of the two lines
        angle0 = left_line[0][6]
        angle1 = right_line[0][6]
        avg_angle = (angle0 + angle1) / 2

        # Calculate new points along this direction
        dx = np.cos(avg_angle)
        dy = np.sin(avg_angle)

        mid_x1, mid_y1 = mid_x, mid_y
        mid_x2, mid_y2 = mid_x - 100 * dx, mid_y - 100 * dy
    else:
        # Find intersection point
        b = np.array([rho0, rho1])
        x_int, y_int = np.linalg.solve(A, b)

        # Find similar unit direction vectors along each line
        d0 = np.array([-np.sin(theta0), np.cos(theta0)])
        if d0[1] > 0:
            d0 = -d0
        d1 = np.array([-np.sin(theta1), np.cos(theta1)])
        if d1[1] > 0:
            d1 = -d1

        # Find the average direction
        d = d0 + d1
        if np.linalg.norm(d) < 1e-8:
            # If directions cancel out, use perpendicular
            d = np.array([d0[1], -d0[0]])
        d /= np.linalg.norm(d)

        # Create points for the midline
        mid_x1, mid_y1 = x_int, y_int
        mid_x2, mid_y2 = x_int + 100 * d[0], y_int + 100 * d[1]

    # Create midline with same format as input
    angle = -np.arctan2(mid_y2 - mid_y1, mid_x2 - mid_x1)
    theta = angle - np.pi / 2
    rho = mid_x1 * np.cos(theta) + mid_y1 * np.sin(theta)

    mid_line = [[mid_x1, mid_y1, mid_x2, mid_y2, rho, theta, angle, 0.0]]

    return np.array(mid_line)

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

def draw_hough_lines(image, lines, color, label_color):
    if lines is not None and len(lines) > 0:  # Ensure lines is not None and has elements
        for line in lines:
            # Unpack the extended line format
            if len(line) == 8:
                # Format [x1, y1, x2, y2, rho, theta, angle, error]
                x1, y1, x2, y2, rho, theta, angle, error = line
            else:
                # Format [x1, y1, x2, y2, rho, theta, angle]
                x1, y1, x2, y2, rho, theta, angle = line

            # Convert coordinates to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw the actual line segment
            cv.line(image, (x1, y1), (x2, y2), color, 2, cv.LINE_AA)

            # Calculate midpoint for label placement
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Convert angle to degrees for display
            angle_degrees = angle * 180 / np.pi

            # Create and place labels
            label = f"a:{int(angle_degrees)}"
            cv.putText(image, label, (mid_x, mid_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv.LINE_AA)

def process_image(image_path, params, output_base_dir, gallery_dir):
    save_dir, image_name = create_save_directory(output_base_dir, image_path)
    save_original_image(image_path, save_dir)

    src_color = cv.imread(image_path)
    if src_color is None:
        return

    color_segmented = mask_top_half(apply_color_segmentation(src_color, params["min_value"], params["max_saturation"]))
    save_image(color_segmented, save_dir, f"{image_name}_cs.png")

    color_segmented_gray = cv.cvtColor(color_segmented, cv.COLOR_BGR2GRAY)
    color_segmented_canny = cv.Canny(color_segmented_gray, 50, 200, None, 3)
    save_image(color_segmented_canny, save_dir, f"{image_name}_cs_canny.png")

    save_filtered_clustered_hough_lines(src_color, color_segmented_canny, "filtered_clustered_color_seg", params, save_dir, image_name, gallery_dir)

def save_filtered_clustered_hough_lines(color_image, canny, ending, params, save_dir, image_name, gallery_dir=None):
    """
    Process image with enhanced lane line detection that:
    1. Filters lines below the middle of the image
    2. Clusters lines by x-intercept
    3. Merges line candidates with similar x-intercepts
    4. Filters by column with specific criteria for left/right lanes
    5. Chooses lines based on the y-point distance from center
    6. Calculates and draws the midline between detected left and right lanes
    """
    cdst = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
    height, width = canny.shape[:2]

    lane_name = os.path.basename(os.path.dirname(save_dir))

    # Add row_mask_lower_threshold if not present
    if "row_mask_lower_threshold" not in params:
        params["row_mask_lower_threshold"] = height * params.get("row_threshold_factor", 0.5)

    # Get the detected lane lines
    left_lines, right_lines = get_left_and_right_lane_candidates(
        canny, params,
        (params["min_angle_left"], params["max_angle_left"]),
        (params["min_angle_right"], params["max_angle_right"])
    )

    # Draw the original lines in light blue (for comparison)
    if len(left_lines) > 0:
        draw_hough_lines(color_image, left_lines, (255, 200, 100), (0, 0, 0))
    if len(right_lines) > 0:
        draw_hough_lines(color_image, right_lines, (255, 200, 100), (0, 0, 0))

    # Filter lines to be below the middle of the image
    filtered_left_lines = filter_line_candidates_by_row_extent(left_lines, params)
    filtered_right_lines = filter_line_candidates_by_row_extent(right_lines, params)

    # Draw filtered lines in yellow
    if len(filtered_left_lines) > 0:
        draw_hough_lines(color_image, filtered_left_lines, (0, 255, 255), (0, 0, 0))
    if len(filtered_right_lines) > 0:
        draw_hough_lines(color_image, filtered_right_lines, (0, 255, 255), (0, 0, 0))

    # Cluster and merge lines
    merged_left_lines = cluster_and_merge_lines(
        filtered_left_lines,
        eps=params.get("x_intercept_cluster_eps", 20),
        min_samples=params.get("x_intercept_min_samples", 1)
    )
    merged_right_lines = cluster_and_merge_lines(
        filtered_right_lines,
        eps=params.get("x_intercept_cluster_eps", 20),
        min_samples=params.get("x_intercept_min_samples", 1)
    )

    # Draw merged lines in orange
    if len(merged_left_lines) > 0:
        draw_hough_lines(color_image, merged_left_lines, (0, 128, 255), (0, 0, 0))
    if len(merged_right_lines) > 0:
        draw_hough_lines(color_image, merged_right_lines, (0, 128, 255), (0, 0, 0))

    # Filter by column extent
    column_filtered_left_lines = filter_line_candidates_by_column_extent(merged_left_lines, width, is_left=True)
    column_filtered_right_lines = filter_line_candidates_by_column_extent(merged_right_lines, width, is_left=False)

    # Draw column filtered lines in purple
    if len(column_filtered_left_lines) > 0:
        draw_hough_lines(color_image, column_filtered_left_lines, (255, 0, 255), (0, 0, 0))
    if len(column_filtered_right_lines) > 0:
        draw_hough_lines(color_image, column_filtered_right_lines, (255, 0, 255), (0, 0, 0))

    # Select the best lines
    best_left_line = select_best_line_candidate(column_filtered_left_lines, width, height)
    best_right_line = select_best_line_candidate(column_filtered_right_lines, width, height)

    # Draw the final selected lines
    if best_left_line is not None:
        draw_hough_lines(color_image, best_left_line, (0, 255, 0), (255, 255, 255)) # green
    if best_right_line is not None:
        draw_hough_lines(color_image, best_right_line, (0, 0, 255), (255, 255, 255)) # red

    # Calculate and draw midline
    mid_line = get_mid_line(best_left_line, best_right_line)
    if mid_line is not None:
        draw_hough_lines(color_image, mid_line, (0, 0, 0), (255, 255, 255)) # orange

    # Save the image
    save_image(color_image, save_dir, f"{image_name}_{ending}.png")

    # Also save to gallery if specified
    if gallery_dir is not None:
        save_image(color_image, gallery_dir, f"{image_name}_{lane_name}_{ending}.png")

def process_and_save_images(params, input_dir, output_base_dir):
    prepare_output_directory(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)  # Ensure the base directory exists
    gallery_dir = os.path.join(output_base_dir, "gallery")
    os.makedirs(gallery_dir)
    save_params(output_base_dir)
    all_image_paths = get_all_image_paths(input_dir)

    for image_path in all_image_paths:
        process_image(image_path, params, output_base_dir, gallery_dir)

params = {
    "rho_resolution": 5,
    "theta_resolution": np.pi / 30,
    "max_saturation": 40.,
    "min_value": 140.,
    "min_angle_left": np.pi/12,
    "max_angle_left": np.pi/2,
    "min_angle_right": np.pi/2,
    "max_angle_right": 11*np.pi/12,
    # "min_angle_left": 0,
    # "max_angle_left": np.pi/2,
    # "min_angle_right": np.pi/2,
    # "max_angle_right": np.pi,
    "threshold": 50,
    # Add required parameters for HoughLinesP
    "minLineLength": 20,  # Minimum length of line segments (in pixels)
    "maxLineGap": 100,    # Maximum allowed gap between line segments (in pixels)
    # Add new parameters for enhanced line detection
    "row_threshold_factor": 0.5,  # Factor to determine row threshold
    "x_intercept_cluster_eps": 30,  # Clustering epsilon parameter
    "x_intercept_min_samples": 1   # Min samples for clustering
}

# angle is gonna be on [-pi, pi], but need to figure out what x0, y0 and x1, y1 are on the image so I can figure out the angle needed accordingly

computer_vision_path = '/home/racecar/monorepo/src/final_challenge2025/race_to_the_moon/race_to_the_moon/computer_vision'
input_dir = os.path.join(computer_vision_path, "racetrack_images")
output_dir = os.path.join(computer_vision_path, "test_results_hough_lines_enhanced_6")

process_and_save_images(params, input_dir, output_dir)
