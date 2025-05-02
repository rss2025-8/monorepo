import cv2 as cv
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
    cv.imshow("image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def cd_color_segmentation(img, max_saturation=25, min_saturation=0., min_value=190., max_value=255.):
    """
    Returns mask of image having all possible hues and specified range of saturation and value
    Input:
    img: np.3darray; the input image with a cone to be detected. BGR.
    Return:
    mask: binary mask highlighting regions with specified saturation and value ranges
    """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Good reference: https://colorizer.org Note cv2 has max saturation and max value = 255 not 100

    mask = cv.inRange(hsv,
                      np.array([0, min_saturation, min_value]),
                      np.array([179, max_saturation, max_value]))

    mask = cv.inRange(hsv,
                      np.array([0, min_saturation, min_value]),
                      np.array([179, max_saturation, max_value]))

    # # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    return mask

def bgr_color_segmentation(bgr_img, min_value, max_saturation):
    mask = cd_color_segmentation(bgr_img, min_value=min_value, max_saturation=max_saturation)
    return cv.bitwise_and(bgr_img, bgr_img, mask=mask)
