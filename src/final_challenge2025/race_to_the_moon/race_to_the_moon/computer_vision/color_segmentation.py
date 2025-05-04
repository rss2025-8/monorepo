"""Contains color_segmentation_white() to find the white portions of an image.

H (0-360), S (0-100), V (0-100).

Must include:
[
[45, 2, 99],
[41.2, 7, 89],
[180, 2, 91],
[86.7, 4, 90],
[132.9, 6, 93],
[77.1, 4, 74],
[17.1, 9, 88],
]

Ideally include:
[
[60, 2, 82],
[33.7, 6, 97],
[126.7, 4, 91],
[51.4, 4, 75],
[110.8, 6, 92],
[9.2, 7, 71],
[45, 9, 71],
]

Must avoid:
[
[184, 10, 56],
[18, 37, 53],
[17.8, 18, 59],
[202.1, 14, 55],
[11.8, 29, 69],
[34.2, 62, 54],
[216.6, 51, 49],
]

Ideally avoid:
[
[150, 7, 72],
[144, 4, 47],
[145.7, 4, 62],
[5, 9, 50],
[4.8, 66, 58],
[355.4, 11, 45],
]

Strict thresholds:
H: [0, 180]
S: [0, 9]
V: [71, 100]

Obstacle thresholds:
H: [0, 360]
S: [4, 100]
V: [0, 72]

Final thresholds:
H: [0, 360] -> [0, 179]
S: [0, 16] -> [0, 41]
V: [64, 100] -> [163, 255]

Keep in mind that the top part of the image is masked out.
It's slightly better to include too much than too little.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################


def image_print(img):
    """
    Helper function to print out images, for debugging. Pass them in as a list.
    Press any key to continue.
    """
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def color_segmentation_white(img, min_saturation=0, max_saturation=41, min_value=163, max_value=255):
    """
    Return a mask of the white portions of an image.
    Input:
    img: np.3darray; the input image. BGR.
    min_saturation: float; the minimum saturation of the white portion.
    max_saturation: float; the maximum saturation of the white portion.
    min_value: float; the minimum value of the white portion.
    max_value: float; the maximum value of the white portion.
    Return:
    mask: binary mask highlighting regions with specified saturation and value ranges
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array([0, min_saturation, min_value]), np.array([179, max_saturation, max_value]))

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # mask = cv2.dilate(mask, kernel)  # For easier line detection

    return mask
