import cv2 as cv
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, TypeVar
from sklearn.cluster import DBSCAN

# def get_left_and_right_lane_candidates(canny, params):
#     left_lines = get_line_candidates(canny, params)
#     right_lines = get_line_candidates(canny, params)
#     return left_lines, right_lines

def get_left_and_right_lane_candidates(canny, params, left_angle_range, right_angle_range):
    lines = get_line_candidates(canny, params)
    if not lines:
        return [], []
    lines_array = np.array(lines)
    left_lines = filter_line_candidates_by_angle(lines_array, left_angle_range)
    right_lines = filter_line_candidates_by_angle(lines_array, right_angle_range)
    return left_lines, right_lines

def get_line_candidates(canny: np.ndarray, params: Dict[str, Any]) -> List[np.ndarray]:
    # min_angle, max_angle = angle_range
    lines = cv.HoughLinesP(canny, rho=params["rho_resolution"], theta=params["theta_resolution"], threshold=params["threshold"], minLineLength=params["minLineLength"], maxLineGap=params["maxLineGap"])
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

    # print(f"{canny.shape=}")
    # mask_top_of_image_threshold = canny.shape[0] // 2
    # row_mask_lower_threshold = params["row_mask_lower_threshold"]

    # Calculate angles for all lines
    # x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]
    # angles = np.arctan2(y2 - y1, x2 - x1)

    # Create a mask for lines within the angle range AND in the bottom half of the image
    # We want y1 >= threshold (below the middle) OR y2 >= threshold
    # mask = (min_angle <= angles) & (angles <= max_angle) & ((y1 >= row_mask_lower_threshold) | (y2 >= row_mask_lower_threshold))

    # Filter lines using the mask
    # filtered_lines = lines[mask]
    # filtered_lines = lines

    # For each filtered line, calculate rho and theta
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
    # filtered_lines = line_candidates
    return filtered_lines
# def filter_line_candidates_by_column_extent(line_candidates, params,  column_range: Tuple[float, float]):
#     min_x, max_x = column_range
#     x1, x2 = line_candidates[:, 0], line_candidates[:, 2]
#     mask = ((max_x >= x1 >= min_x) | (max_x >= x2 >= min_x))
#     filtered_lines = line_candidates[mask]
#     return filtered_lines

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

# def select_best_line_candidate(line_candidates, image_width, image_height):
#     if len(line_candidates) == 0:
#         return None

#     # Get the point with the maximum y value for each line
#     y1, y2 = line_candidates[:, 1], line_candidates[:, 3]
#     max_y_values = np.maximum(y1, y2)

#     # Select the line with the maximum y value
#     best_idx = np.argmax(max_y_values)
#     return line_candidates[best_idx:best_idx+1]

def select_best_line_candidate(line_candidates, image_width, image_height):
    if len(line_candidates) == 0:
        return None

    # Get the point with the maximum y value for each line
    y2 = line_candidates[:, 3]

    # Select the line with the maximum y value
    best_idx = np.argmax(y2)
    return line_candidates[best_idx:best_idx+1]

def select_best_line_candidate_by_midpoint_radius(line_candidates, image_width, image_height):
    if len(line_candidates) == 0:
        return None

    best_line = get_nearest_line(np.array([image_width/2, image_height]), line_candidates)

    return best_line

def mask_top_half(image: np.ndarray) -> np.ndarray:
    """
    Masks the top half of the image by setting all pixels with y < image height / 2 to black.
    """
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[height // 2:, :] = 255  # Keep the bottom half
    return cv.bitwise_and(image, image, mask=mask)

def get_nearest_line(P: np.ndarray, Lines: np.ndarray) -> np.ndarray:
    """Return the line nearest the point P. Lines is N by 8. P is (x, y)."""
    if Lines.shape[0] == 0:
        raise ValueError("Lines array is empty. Ensure valid input is provided.")
    Ps = np.tile(P, (Lines.shape[0], 1))  # N x 2, N (# of segments) copies of car_loc
    dists = vectorized_point_to_segment_distance(Ps, Lines)
    return Lines[np.argmin(dists):np.argmin(dists)+1]

def vectorized_point_to_segment_distance(P: np.ndarray, Lines: np.ndarray) -> np.ndarray:
    """Returns the minimum distance from point P[i] to the Line (x1[i],y1[i],x2[i],y2[i],...) for all i.

    P should be N x 2, Lines should be N x 8. Returns a 1D array of length N.
    """
    S1, S2 = Lines[:, :2], Lines[:, 2:4]
    assert P.shape[0] == S1.shape[0] == S2.shape[0]
    assert P.shape[1] == 2 and S1.shape[1] == 2 and S2.shape[1] == 2

    diff = S2 - S1  # N x 2
    L2 = np.sum(diff * diff, axis=1)  # N
    L2 = np.where(L2 > 0, L2, 1.0)  # N (Avoid division by zero)

    # Projection of p onto s1s2 for each segment
    t = np.sum((P - S1) * diff, axis=1) / L2
    t = np.clip(t, 0.0, 1.0)  # N
    # Closest points on segments
    projection = S1 + diff * t[:, None]  # N x 2
    # Distances to closest points
    return np.linalg.norm(P - projection, axis=1)
