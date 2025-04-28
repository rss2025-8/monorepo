import cv2 as cv
import numpy as np
import os
import math
import shutil
from pathlib import Path
from color_segmentation import cd_color_segmentation

def process_and_save_images(params, input_dir, output_base_dir):
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)

    rho_resolution = params["rho_resolution"]
    theta_resolution = params["theta_resolution"]
    max_saturation = params["max_saturation"]
    min_value = params["min_value"]
    min_angle_left = params["min_angle_left"]
    max_angle_left = params["max_angle_left"]
    min_angle_right = params["min_angle_right"]
    max_angle_right = params["max_angle_right"]
    threshold = params["threshold"]

    all_image_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(root, file))

    for image_path in all_image_paths:
        lane_name = os.path.basename(os.path.dirname(image_path))
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        save_dir = os.path.join(output_base_dir, lane_name, image_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        orig_image_path = os.path.join(save_dir, os.path.basename(image_path))
        shutil.copy(image_path, orig_image_path)

        # Read image in color format
        src_color = cv.imread(image_path)
        if src_color is None:
            continue

        # Create grayscale version for edge detection
        src = cv.cvtColor(src_color, cv.COLOR_BGR2GRAY)

        dst = cv.Canny(src, 50, 200, None, 3)
        cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)

        # Apply color segmentation to the color image
        mask = cd_color_segmentation(src_color, min_value=min_value, max_saturation=max_saturation)

        # Apply mask to grayscale image
        color_segmented_src = cv.bitwise_and(src, src, mask=mask)

        color_segmented_path = os.path.join(save_dir, f"{image_name}_color_segmented.png")
        cv.imwrite(color_segmented_path, color_segmented_src)
        color_segmented_img = cv.cvtColor(color_segmented_src, cv.COLOR_GRAY2BGR)

        canny_path = os.path.join(save_dir, f"{image_name}_canny.png")
        cv.imwrite(canny_path, dst)

        left_line_candidates = cv.HoughLines(dst, rho_resolution, theta_resolution, threshold, min_theta=min_angle_left, max_theta=max_angle_left)
        right_line_candidates = cv.HoughLines(dst, rho_resolution, theta_resolution, threshold, min_theta=min_angle_right, max_theta=max_angle_right)

        # Initialize length variables to avoid reference errors
        length_left_line_candidates = 0
        length_right_line_candidates = 0

        if left_line_candidates is not None:
            length_left_line_candidates = len(left_line_candidates)
            for i in range(0, length_left_line_candidates):
                rho = left_line_candidates[i][0][0]
                theta = left_line_candidates[i][0][1]
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

                label = f"{int(theta_degrees)}"
                cv.putText(cdst, label, (mid_x, mid_y), cv.FONT_HERSHEY_SIMPLEX,
                           0.25, (0, 255, 0), 1, cv.LINE_AA)

        if right_line_candidates is not None:
            length_right_line_candidates = len(right_line_candidates)
            for i in range(0, length_right_line_candidates):
                rho = right_line_candidates[i][0][0]
                theta = right_line_candidates[i][0][1]
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

                label = f"{int(theta_degrees)}"
                cv.putText(cdst, label, (mid_x, mid_y), cv.FONT_HERSHEY_SIMPLEX,
                           0.25, (0, 255, 0), 1, cv.LINE_AA)

        standard_path = os.path.join(save_dir, f"{image_name}_standard.png")
        cv.imwrite(standard_path, cdst)

        linesP = cv.HoughLinesP(dst, rho_resolution, theta_resolution, threshold, minLineLength=50, maxLineGap=10)

        # Initialize length_lines to avoid reference errors
        length_lines = 0

        if linesP is not None:
            length_lines = len(linesP)
            for i in range(0, length_lines):
                l = linesP[i][0]
                if l[2] != l[0]:
                    angle = math.atan2(l[3] - l[1], l[2] - l[0])
                    theta_degrees = angle * 180 / np.pi

                    cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 2, cv.LINE_AA)

                    mid_x = (l[0]+l[2]) // 2
                    mid_y = (l[1]+l[3]) // 2
                    label = f"{int(theta_degrees)}"
                    cv.putText(cdstP, label, (mid_x, mid_y), cv.FONT_HERSHEY_SIMPLEX,
                               0.25, (0, 255, 0), 1, cv.LINE_AA)

        probabilistic_path = os.path.join(save_dir, f"{image_name}_probabilistic.png")
        cv.imwrite(probabilistic_path, cdstP)

        color_segmented_left_line_candidates = cv.HoughLines(color_segmented_src, rho_resolution, theta_resolution, threshold, min_theta=min_angle_left, max_theta=max_angle_left)
        color_segmented_right_line_candidates = cv.HoughLines(color_segmented_src, rho_resolution, theta_resolution, threshold, min_theta=min_angle_right, max_theta=max_angle_right)

        if color_segmented_left_line_candidates is not None:
            length_left_line_candidates = len(color_segmented_left_line_candidates)
            for i in range(0, length_left_line_candidates):
                rho = color_segmented_left_line_candidates[i][0][0]
                theta = color_segmented_left_line_candidates[i][0][1]
                theta_degrees = theta * 180 / np.pi

                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

                cv.line(color_segmented_img, pt1, pt2, (0,0,255), 2, cv.LINE_AA)

                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2

                label = f"{int(theta_degrees)}"
                cv.putText(color_segmented_img, label, (mid_x, mid_y), cv.FONT_HERSHEY_SIMPLEX,
                           0.25, (0, 255, 0), 1, cv.LINE_AA)

        if color_segmented_right_line_candidates is not None:
            length_right_line_candidates = len(color_segmented_right_line_candidates)
            for i in range(0, length_right_line_candidates):
                rho = color_segmented_right_line_candidates[i][0][0]
                theta = color_segmented_right_line_candidates[i][0][1]
                theta_degrees = theta * 180 / np.pi

                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

                cv.line(color_segmented_img, pt1, pt2, (0,0,255), 2, cv.LINE_AA)

                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2

                label = f"{int(theta_degrees)}"
                cv.putText(color_segmented_img, label, (mid_x, mid_y), cv.FONT_HERSHEY_SIMPLEX,
                           0.25, (0, 255, 0), 1, cv.LINE_AA)

        color_segmented_hough_lines_path = os.path.join(save_dir, f"{image_name}_color_segmented_hough_lines.png")
        cv.imwrite(color_segmented_hough_lines_path, color_segmented_img)

    return 0

params = {
    "rho_resolution": 1,
    "theta_resolution": np.pi / 180,
    "max_saturation": 25.,
    "min_value": 190.,
    "min_angle_left": np.pi/4,
    "max_angle_left": np.pi/2,
    "min_angle_right": 0,
    "max_angle_right": np.pi/4,
    "threshold": 150
}

computer_vision_path = '/home/racecar/monorepo/src/final_challenge2025/race_to_the_moon/race_to_the_moon/computer_vision'

input_dir = os.path.join(computer_vision_path, "racetrack_images")
output_dir = os.path.join(computer_vision_path, "test_results_hough_lines")

process_and_save_images(params, input_dir, output_dir)
