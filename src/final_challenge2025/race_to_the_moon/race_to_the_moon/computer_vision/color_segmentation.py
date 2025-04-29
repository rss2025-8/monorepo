"""Contains color_segmentation_white() to find the white portions of an image."""

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


def color_segmentation_white(img, max_saturation=25, min_saturation=0.0, min_value=190.0, max_value=255.0):
    """
    Return a mask of the white portions of an image.
    Input:
    img: np.3darray; the input image. BGR.
    max_saturation: float; the maximum saturation of the white portion.
    min_saturation: float; the minimum saturation of the white portion.
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

    return mask
