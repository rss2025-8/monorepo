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
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def image_print(img):
    """
    Helper function to print out images, for debugging. Pass them in as a list.
    Press any key to continue.
    """
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def light_is_green(img):
    """
    img: np.3darray; the input image with a cone to be detected. BGR.
    Returns True if a green light is visible, False otherwise.
    """
    # Set top third of image to black
    img[: img.shape[0] // 3, :, :] = 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # values i got from an hsv color picker online:
    # 160, 100, 80
    # 170, 100, 50
    # math inside np converts from hsv color space to cv hsv color space
    # lower_color = np.array([160 / 2, 100 * 2.55, 70 * 2.55])  # lower-bound value
    # upper_color = np.array([170 / 2, 100 * 2.55, 80 * 2.55])  # upper-bound value

    # lower_color = np.array([79, 132, 153])  # lower-bound value
    lower_color = np.array([79, 132, 140])  # lower-bound value
    # upper_color = np.array([82, 145, 255])  # upper-bound value
    upper_color = np.array([93, 255, 255])  # upper-bound value
    # upper_color = np.array([82, 255, 255])  # upper-bound value

    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Apply an opening, followed by a closing operation to remove noise and fill in small holes
    kernel = np.ones((9, 9), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # displays in rviz
    # image_print(img)
    # image_print(mask)
    # image_print(cleaned_mask)

    # Check contour area
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # print(cv2.contourArea(largest_contour))
        if cv2.contourArea(largest_contour) > 15:
            return True
    return False
