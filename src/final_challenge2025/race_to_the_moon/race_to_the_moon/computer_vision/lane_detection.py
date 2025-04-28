import cv2 as cv
import numpy as np
import math

def get_lane_pxs(hough_line_params, image):
    assert len(hough_line_params) == 3
    hough_method, left_lane_angle_range, right_lane_angle_range = hough_line_params
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    # Canny edge detection
    edges = cv.Canny(gray, 50, 200, None, 3)

    # degrees to radians
    left_min, left_max = np.radians(left_lane_angle_range[0]), np.radians(left_lane_angle_range[1])
    right_min, right_max = np.radians(right_lane_angle_range[0]), np.radians(right_lane_angle_range[1])

    left_line = None
    right_line = None

    if hough_method == "standard":
        lines = cv.HoughLines(edges, 1, np.pi / 180, 100, None, 0, 0)
        if lines is not None:
            left_candidates = []
            right_candidates = []

            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]

                # Normalize theta on [-π/2 and π/2]
                while theta > np.pi/2:
                    theta -= np.pi
                while theta < -np.pi/2:
                    theta += np.pi

                if left_min <= theta <= left_max:
                    left_candidates.append((rho, theta))
                elif right_min <= theta <= right_max:
                    right_candidates.append((rho, theta))

            # Find best candidate for lanes
            if left_candidates:
                # Strongest line typically first in the list for HoughLines
                rho, theta = left_candidates[0]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                left_line = (pt1[0], pt1[1], pt2[0], pt2[1])

            if right_candidates:
                # Choose the strongest line
                rho, theta = right_candidates[0]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                right_line = (pt1[0], pt1[1], pt2[0], pt2[1])

    elif hough_method == "probabilistic":
        linesP = cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
        if linesP is not None:
            left_candidates = []
            right_candidates = []

            for i in range(0, len(linesP)):
                l = linesP[i][0]

                # Calculate angle
                if l[2] != l[0]:  # Avoid division by zero
                    angle = math.atan2(l[3] - l[1], l[2] - l[0])

                    # Normalize angle to be between -π/2 and π/2
                    while angle > np.pi/2:
                        angle -= np.pi
                    while angle < -np.pi/2:
                        angle += np.pi

                    if left_min <= angle <= left_max:
                        left_candidates.append(l)
                    elif right_min <= angle <= right_max:
                        right_candidates.append(l)

            if left_candidates:
                left_line = left_candidates[0]

            if right_candidates:
                right_line = right_candidates[0]

    return left_line, right_line

def get_lane_rho_thetas(hough_line_params, image):
    assert len(hough_line_params) == 3
    hough_method, left_lane_angle_range, right_lane_angle_range = hough_line_params

    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    # Canny edge detection
    edges = cv.Canny(gray, 50, 200, None, 3)

    # Convert degrees to radians
    left_min, left_max = np.radians(left_lane_angle_range[0]), np.radians(left_lane_angle_range[1])
    right_min, right_max = np.radians(right_lane_angle_range[0]), np.radians(right_lane_angle_range[1])

    left_rho_theta = None
    right_rho_theta = None

    if hough_method == "standard":
        lines = cv.HoughLines(edges, 1, np.pi / 180, 100, None, 0, 0)

        if lines is not None:
            left_candidates = []
            right_candidates = []

            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]

                # Normalize theta on [-π/2 and π/2]
                while theta > np.pi/2:
                    theta -= np.pi
                while theta < -np.pi/2:
                    theta += np.pi

                if left_min <= theta <= left_max:
                    left_candidates.append((rho, theta))
                elif right_min <= theta <= right_max:
                    right_candidates.append((rho, theta))

            # Find best candidate for lanes
            if left_candidates:
                # Strongest line typically first in the list for HoughLines
                left_rho_theta = left_candidates[0]

            if right_candidates:
                # Choose the strongest line
                right_rho_theta = right_candidates[0]

    elif hough_method == "probabilistic":
        linesP = cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)

        if linesP is not None:
            left_candidates = []
            right_candidates = []

            for i in range(0, len(linesP)):
                l = linesP[i][0]

                # Calculate angle
                if l[2] != l[0]:  # Avoid division by zero
                    angle = math.atan2(l[3] - l[1], l[2] - l[0])

                    # Calculate rho (perpendicular distance from origin to line)
                    # Using formula rho = x*cos(theta) + y*sin(theta)
                    # Using midpoint of line segment
                    mid_x = (l[0] + l[2]) / 2
                    mid_y = (l[1] + l[3]) / 2
                    rho = mid_x * math.cos(angle) + mid_y * math.sin(angle)

                    # Normalize angle to be between -π/2 and π/2
                    while angle > np.pi/2:
                        angle -= np.pi
                        rho = -rho  # Adjust rho when normalizing theta
                    while angle < -np.pi/2:
                        angle += np.pi
                        rho = -rho  # Adjust rho when normalizing theta

                    if left_min <= angle <= left_max:
                        left_candidates.append((rho, angle))
                    elif right_min <= angle <= right_max:
                        right_candidates.append((rho, angle))

            if left_candidates:
                left_rho_theta = left_candidates[0]
            if right_candidates:
                right_rho_theta = right_candidates[0]

    return left_rho_theta, right_rho_theta

def rho_theta_to_pxs(rho_theta):
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
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

    # Return the line endpoints in pixel coordinates
    return (pt1[0], pt1[1], pt2[0], pt2[1])
