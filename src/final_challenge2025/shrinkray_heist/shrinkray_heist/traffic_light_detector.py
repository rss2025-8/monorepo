"""
(Green)
Include:
[
[163.1, 53, 96],
[157.6, 54, 66],
[159.5, 56, 62],
[183, 100, 68],
[164.3, 98, 51],
[166.6, 100, 26],
[166.3, 99, 47],
]

Probably include:
[
[180, 1, 100],
]

Avoid: Everything else

Bounds:
H: [158, 186] -> [79, 93]
S: [52, 100] -> [132, 255]
V: [55, 100] -> [140, 255]

(Red)
Include:
[
[3.7, 96, 100],
[0, 99, 49],
[0, 95, 59],
[359, 97, 69],
[4.4, 92, 52],
]

Avoid: Everything else

Bounds:
H: [0, 6] and [358, 360]
S: [88, 100]
V: [50, 100]
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

DEBUG = False


def image_print(img):
    """
    Helper function to print out images, for debugging. Pass them in as a list.
    Press any key to continue.
    """
    if DEBUG:
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def light_is_green(img):
    """
    img: np.3darray; the input image with a cone to be detected. BGR.
    Returns True if a green light is visible, False otherwise.
    """
    # Set top 0.35 of image to black to avoid exit sign
    img[: int(img.shape[0] * 0.35), :, :] = 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_color = np.array([79, 132, 140])  # lower-bound value
    upper_color = np.array([93, 255, 255])  # upper-bound value

    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Apply an opening, followed by a closing operation to remove noise and fill in small holes
    kernel = np.ones((9, 9), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # displays in rviz
    image_print(img)
    image_print(mask)
    image_print(cleaned_mask)

    # Check contour area
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # print(cv2.contourArea(largest_contour))
        # Draw box around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        image_print(img)
        if cv2.contourArea(largest_contour) > 20:
            return True
    return False


def light_is_red(img):
    """
    img: np.3darray; the input image with a cone to be detected. BGR.
    Returns True if a red light is visible, False otherwise.
    """
    # Set top 0.35 of image to black to avoid exit sign
    img[: int(img.shape[0] * 0.35), :, :] = 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    image_print(img)

    # math inside np converts from hsv color space to cv hsv color space
    lower_color = np.array([0 / 2, 88 * 2.55, 50 * 2.55])  # lower-bound value
    upper_color = np.array([6 / 2, 100 * 2.55, 100 * 2.55])  # upper-bound value

    # Also get near 360
    lower_color_360 = np.array([358 / 2, 88 * 2.55, 50 * 2.55])
    upper_color_360 = np.array([360 / 2, 100 * 2.55, 100 * 2.55])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask_360 = cv2.inRange(hsv, lower_color_360, upper_color_360)
    mask = cv2.bitwise_or(mask, mask_360)
    # Apply an opening, followed by a closing operation to remove noise and fill in small holes
    kernel = np.ones((9, 9), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # displays in rviz
    image_print(mask)
    image_print(cleaned_mask)

    # Check contour area
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # print(cv2.contourArea(largest_contour))
        # Draw box around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        image_print(img)
        if cv2.contourArea(largest_contour) > 20:
            return True
    return False
