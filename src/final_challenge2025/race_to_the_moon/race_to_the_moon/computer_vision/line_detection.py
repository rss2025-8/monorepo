"""Contains helper functions for detecting lines in an image."""

import math

import cv2 as cv
import numpy as np


def circular_mean(thetas):
    """Returns the circular mean of a list of angles."""
    sin_sum = np.sum(np.sin(thetas))
    cos_sum = np.sum(np.cos(thetas))
    return np.arctan2(sin_sum, cos_sum)


def get_raw_lines(hough_line_params, image):
    """Returns a list of tuples (rho, theta) for all detected lines, using a Hough Transform.

    Rho is the perpendicular distance from the origin to the line.
    Theta is the angle between the x-axis and the line.
    """
    assert len(hough_line_params) == 3
    hough_method, _, _ = hough_line_params

    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    # Canny edge detection
    edges = cv.Canny(gray, 50, 200, None, 3)
    rho_thetas = []

    if hough_method == "standard":
        lines = cv.HoughLines(edges, 1, np.pi / 180, 100, None, 0, 0)
        rho_thetas = [(rho, theta) for rho, theta in lines[0]]
    elif hough_method == "probabilistic":
        linesP = cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
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


def get_best_lanes(hough_line_params, image):
    """Returns two tuples of (rho, theta) for the left and right lanes, or None if a given lane is not detected."""

    left_rho_thetas, right_rho_thetas = get_all_lanes(hough_line_params, image)

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


def rho_theta_to_pxs(rho_theta):
    """Returns a tuple of (x0, y0, x1, y1) for the line endpoints in pixel coordinates."""
    if rho_theta is None:
        return None

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
    return (pt1[0], pt1[1], pt2[0], pt2[1])
