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
  
def cd_color_segmentation(img):
    """
    img: np.3darray; the input image with a cone to be detected. BGR.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # values i got from an hsv color picker online:
    # 160, 100, 80
    # 170, 100, 50
    # math inside np converts from hsv color space to cv hsv color space
    upper_color = np.array([170/2, 100*2.55, 80*2.55]) #upper-bound value
    lower_color = np.array([160/2, 100*2.55, 70*2.55]) #lower-bound value

    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # displays in rviz
    image_print(img)
    image_print(mask)

    return bool(contours)

