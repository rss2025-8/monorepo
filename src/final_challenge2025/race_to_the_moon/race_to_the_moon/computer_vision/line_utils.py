"""
Contains helper functions for detecting lines in an image.

Lines can be represented by rho_theta or endpoints, and in uv or xy coordinates.
"""

import math

import cv2 as cv
import numpy as np
from race_to_the_moon import homography


def find_edges(image, canny_threshold1=50, canny_threshold2=150):
    """Returns a binary edge map of the image."""
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, canny_threshold1, canny_threshold2, None, 3)
    # Dilate edges to fill in gaps (produces more duplicate lines)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv.dilate(edges, kernel)
    return edges


def get_raw_lines(hough_line_params, image):
    """Returns a list of tuples (rho, theta) for all detected lines, using a Hough Transform.

    Rho is the perpendicular distance from the origin to the line.
    Theta is the angle between the x-axis and the line.
    """
    assert len(hough_line_params) == 3
    hough_method, _, _ = hough_line_params
    edges = find_edges(image)
    rho_thetas = []

    # TODO Parameters for HoughLines and HoughLinesP (rho and theta are resolutions)
    if hough_method == "standard":
        lines = cv.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=100)
        if lines is not None:
            rho_thetas = [(rho, theta) for rho, theta in lines[0]]
    elif hough_method == "probabilistic":
        linesP = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        if linesP is not None:
            for x1, y1, x2, y2 in linesP[:, 0]:
                # Direction vector
                dx, dy = x2 - x1, y2 - y1
                # Angle of normal to the line
                theta = np.arctan2(dy, dx) + np.pi / 2
                # Constrain to be in (-pi/2, pi/2]
                theta = theta % np.pi
                if theta > np.pi / 2:
                    theta -= np.pi
                # Distance from origin
                rho = x1 * np.cos(theta) + y1 * np.sin(theta)
                rho_thetas.append((rho, theta))
    else:
        raise ValueError(f"Invalid Hough method: {hough_method}")

    return rho_thetas


def circular_mean(thetas):
    """Returns the circular mean of a list of angles."""
    sin_sum = np.sum(np.sin(thetas))
    cos_sum = np.sum(np.cos(thetas))
    return np.arctan2(sin_sum, cos_sum)


def normalize_endpoints(endpoints, amount_in_front=5.0):
    """
    Extends the line to make the returned endpoints have x1 = 0 and x2 = amount_in_front.
    """
    assert endpoints.shape == (2, 2)
    x0, y0 = endpoints[0]
    x1, y1 = endpoints[1]
    # Ensure x0 < x1
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    # Calculate point on line that has x = 0
    dx = x1 - x0
    if abs(dx) < 1e-6:
        dx = 1e-6  # Avoid division by zero
    t = -x0 / dx
    new_x0, new_y0 = 0.0, y0 + t * (y1 - y0)
    # Calculate point on line that has x = amount_in_front
    t = (amount_in_front - x0) / dx
    new_x1, new_y1 = amount_in_front, y0 + t * (y1 - y0)
    return np.array([[new_x0, new_y0], [new_x1, new_y1]])


def rho_theta_to_endpoints(rho_theta, amount_in_front=5.0):
    """Returns a 2D numpy array of [[x1, y1], [x2, y2]] for the line endpoints in pixel coordinates.

    Guarantees that x1 = 0 and x2 = amount_in_front.
    """
    assert (type(rho_theta) == tuple and len(rho_theta) == 2) or rho_theta.shape == (2,)
    rho, theta = rho_theta

    # Convert polar coordinates to Cartesian form
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho

    # Extend the line to reach image borders (using 1000 as a large value)
    # This creates points far enough to be outside most images
    pt1 = (x0 + 1000 * (-b), y0 + 1000 * (a))
    pt2 = (x0 - 1000 * (-b), y0 - 1000 * (a))

    # Return the line endpoints
    raw_line = np.array([[pt1[0], pt1[1]], [pt2[0], pt2[1]]])
    return normalize_endpoints(raw_line, amount_in_front)


def endpoints_to_rho_theta(endpoints):
    """Returns a tuple of (rho, theta) for the line endpoints as a numpy array of [[x1, y1], [x2, y2]].

    Theta is guaranteed to be in (-pi/2, pi/2].
    """
    assert endpoints.shape == (2, 2)
    x1, y1 = endpoints[0]
    x2, y2 = endpoints[1]
    # Direction vector
    dx, dy = x2 - x1, y2 - y1
    # Angle of normal to the line
    theta = np.arctan2(dy, dx) + np.pi / 2
    # Constrain to be in (-pi/2, pi/2]
    theta = theta % np.pi
    if theta > np.pi / 2:
        theta -= np.pi
    # Distance from origin
    rho = x1 * np.cos(theta) + y1 * np.sin(theta)
    return rho, theta


def find_midline_rho_theta(left_rho_theta, right_rho_theta, is_in_uv=False):
    """Given two lines (rho_i, theta_i), return the midline (rho, theta) with theta in [-pi/2, pi/2].
    is_in_uv is a boolean flag that indicates whether the input lines are in image (uv) coordinates.

    Lines are defined by rho = x * cos(theta) + y * sin(theta).
    This function finds the line that crosses the intersection point of the two lines,
    and has an averaged direction of the two lines.
    Assumes the lines are not parallel!
    """
    assert (type(left_rho_theta) == tuple and len(left_rho_theta) == 2) or left_rho_theta.shape == (2,)
    assert (type(right_rho_theta) == tuple and len(right_rho_theta) == 2) or right_rho_theta.shape == (2,)
    rho0, theta0 = left_rho_theta
    rho1, theta1 = right_rho_theta
    # Find intersection point of the two lines
    A = np.array([[np.cos(theta0), np.sin(theta0)], [np.cos(theta1), np.sin(theta1)]])
    b = np.array([rho0, rho1])
    x_int, y_int = np.linalg.solve(A, b)

    # Find similar unit direction vectors along each line, ensuring they both point upwards in the car frame
    d0 = np.array([-np.sin(theta0), np.cos(theta0)])
    d1 = np.array([-np.sin(theta1), np.cos(theta1)])
    if is_in_uv:
        if d0[1] > 0:
            d0 = -d0
        if d1[1] > 0:
            d1 = -d1
    else:
        if d0[0] < 0:
            d0 = -d0
        if d1[0] < 0:
            d1 = -d1

    # Find the average direction
    d = d0 + d1

    # Special case: Lines are near parallel
    if abs(np.linalg.det(A)) < 1e-8 or abs(np.linalg.norm(d)) < 1e-8:
        # Just average the two lines
        theta = circular_mean([theta0, theta1])
        rho = (rho0 + rho1) / 2
    else:
        # Compute rho, theta for the midline
        d /= np.linalg.norm(d)  # [-sin(theta), cos(theta)]
        theta = np.arctan2(-d[0], d[1])
        rho = x_int * np.cos(theta) + y_int * np.sin(theta)

    return rho, theta


def endpoints_uv_to_xy(endpoints_uv, amount_in_front=5.0):
    """
    Transforms a line from uv (2x2 numpy array of endpoints [[u1, v1], [u2, v2]]) to xy.
    Returns a 2x2 numpy array with A[0] = (x1, y1), A[1] = (x2, y2).
    Guarantees that x1 = 0 and x2 = amount_in_front.
    """
    u0, v0 = endpoints_uv[0][0], endpoints_uv[0][1]
    u1, v1 = endpoints_uv[1][0], endpoints_uv[1][1]
    x0, y0 = homography.transform_uv_to_xy(u0, v0)
    x1, y1 = homography.transform_uv_to_xy(u1, v1)
    return normalize_endpoints(np.array([[x0, y0], [x1, y1]]), amount_in_front)


def get_all_lanes(hough_line_params, image):
    """Returns two lists of tuples of (rho, theta) for all detected left and right lanes. Uses probabilistic."""
    assert len(hough_line_params) == 3
    _, left_lane_angle_range, right_lane_angle_range = hough_line_params

    rho_thetas = get_raw_lines(hough_line_params, image)

    # Convert degrees to radians
    left_min, left_max = np.radians(left_lane_angle_range[0]), np.radians(left_lane_angle_range[1])
    right_min, right_max = np.radians(right_lane_angle_range[0]), np.radians(right_lane_angle_range[1])

    left_rho_thetas = []
    right_rho_thetas = []

    for rho, theta in rho_thetas:
        if left_min <= theta <= left_max:
            left_rho_thetas.append((rho, theta))
        elif right_min <= theta <= right_max:
            right_rho_thetas.append((rho, theta))
    return left_rho_thetas, right_rho_thetas


def filter_close_lanes(left_rho_thetas, right_rho_thetas, last_left_line, last_right_line):
    """
    Filters the left and right lines to be close to the last left and right lines.
    Returns filtered versions of left_rho_thetas and right_rho_thetas, which are lists of (rho, theta) tuples.
    """
    # TODO constants
    # rho_threshold = 10
    # theta_threshold = np.pi / 8
    angle_weight = 16
    num_lines = 4

    def angle_diff(theta1, theta2):
        return min(min(abs(theta1 - theta2), abs((theta1 + np.pi) - theta2)), abs((theta1 - np.pi) - theta2))

    # Score each line and take the best few
    if last_left_line:
        left_scores = [
            (
                abs(rho_theta[0] - last_left_line[0]) + angle_weight * angle_diff(rho_theta[1], last_left_line[1]),
                rho_theta,
            )
            for rho_theta in left_rho_thetas
        ]
        left_scores.sort(key=lambda x: x[0])
        left_rho_thetas = [rho_theta for _, rho_theta in left_scores[:num_lines]]

    if last_right_line:
        right_scores = [
            (
                abs(rho_theta[0] - last_right_line[0]) + angle_weight * angle_diff(rho_theta[1], last_right_line[1]),
                rho_theta,
            )
            for rho_theta in right_rho_thetas
        ]
        right_scores.sort(key=lambda x: x[0])
        right_rho_thetas = [rho_theta for _, rho_theta in right_scores[:num_lines]]

    # if last_left_line:
    #     filtered_left_rho_thetas = [
    #         rho_theta
    #         for rho_theta in left_rho_thetas
    #         if abs(rho_theta[0] - last_left_line[0]) < rho_threshold
    #         and angle_diff(rho_theta[1], last_left_line[1]) < theta_threshold
    #     ]
    #     if filtered_left_rho_thetas:  # Only apply this if any lines match this
    #         left_rho_thetas = filtered_left_rho_thetas

    # if last_right_line:
    #     filtered_right_rho_thetas = [
    #         rho_theta
    #         for rho_theta in right_rho_thetas
    #         if abs(rho_theta[0] - last_right_line[0]) < rho_threshold
    #         and angle_diff(rho_theta[1], last_right_line[1]) < theta_threshold
    #     ]
    #     if filtered_right_rho_thetas:  # Only apply this if any lines match this
    #         right_rho_thetas = filtered_right_rho_thetas

    return left_rho_thetas, right_rho_thetas


def get_best_lanes(hough_line_params, image, last_left_line=None, last_right_line=None):
    """Returns two tuples of (rho, theta) for the left and right lanes, or None if a given lane is not detected.

    If given previous lines, tries to find lines that are close to them as a heuristic.
    """

    left_rho_thetas, right_rho_thetas = get_all_lanes(hough_line_params, image)
    left_rho_thetas, right_rho_thetas = filter_close_lanes(
        left_rho_thetas, right_rho_thetas, last_left_line, last_right_line
    )

    left_line = None
    right_line = None

    if left_rho_thetas:
        # Take mean of rho
        left_rho = np.mean([rho for rho, _ in left_rho_thetas])
        # Circular mean of thetas
        left_thetas = np.array([theta for _, theta in left_rho_thetas])
        left_theta = circular_mean(left_thetas)
        # Record line
        left_line = (left_rho, left_theta)

    # Same for the right line
    if right_rho_thetas:
        right_rho = np.mean([rho for rho, _ in right_rho_thetas])
        right_thetas = np.array([theta for _, theta in right_rho_thetas])
        right_theta = circular_mean(right_thetas)
        right_line = (right_rho, right_theta)

    return left_line, right_line
