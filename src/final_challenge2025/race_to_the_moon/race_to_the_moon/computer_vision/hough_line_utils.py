import cv2 as cv
import numpy as np
import os
import math

def process_and_save_images(input_dir, output_base_dir):
    """
    Process images in input_dir with Canny and Hough transforms
    Save to output_base_dir/subdirectories
    """
    canny_dir = os.path.join(output_base_dir, "canny")
    standard_dir = os.path.join(output_base_dir, "standard")
    probabilistic_dir = os.path.join(output_base_dir, "probabilistic")

    for directory in [canny_dir, standard_dir, probabilistic_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        base_name = os.path.splitext(image_file)[0]

        # Load image
        src = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
        if src is None:
            print(f'Error opening image: {input_path}')
            continue

        # Canny edge detection
        dst = cv.Canny(src, 50, 200, None, 3)
        cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)

        canny_path = os.path.join(canny_dir, f"{base_name}_canny.png")
        cv.imwrite(canny_path, dst)

        # Hough Transform
        standard_path = os.path.join(standard_dir, f"{base_name}_standard.png")
        lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                theta_degrees = theta * 180 / np.pi

                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

                cv.line(cdst, pt1, pt2, (0,0,255), 2, cv.LINE_AA)

                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2

                label = f"{theta_degrees:.1f}°"
                cv.putText(cdst, label, (mid_x, mid_y), cv.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 1, cv.LINE_AA)

        cv.imwrite(standard_path, cdst)

        # Probabilistic Hough Transform
        probabilistic_path = os.path.join(probabilistic_dir, f"{base_name}_probabilistic.png")
        linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                # Calculate angle
                if l[2] != l[0]:
                    angle = math.atan2(l[3] - l[1], l[2] - l[0])
                    theta_degrees = angle * 180 / np.pi

                    cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 2, cv.LINE_AA)

                    # Add theta value as text
                    mid_x = (l[0] + l[2]) // 2
                    mid_y = (l[1] + l[3]) // 2
                    label = f"{theta_degrees:.1f}°"
                    cv.putText(cdstP, label, (mid_x, mid_y), cv.FONT_HERSHEY_SIMPLEX,
                               0.5, (0, 255, 0), 1, cv.LINE_AA)

        cv.imwrite(probabilistic_path, cdstP)

    print(f"Processed {len(image_files)} images. Results saved to {output_base_dir}")
    return 0

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
