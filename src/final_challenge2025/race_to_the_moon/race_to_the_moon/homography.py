"""Helper functions for homography.
transform_uv_to_xy(u, v), transform_xy_to_uv(x, y).
"""

import cv2
import numpy as np

# In inches from the center of the left camera eye
PTS_GROUND_PLANE = [[36, -24], [36, 24], [84, -24], [84, 0], [84, 24], [120, -36], [120, 36]]
# In image pixel coordinates
PTS_IMAGE_PLANE = [
    [537.0, 249.0],
    [99.0, 252.0],
    [417.0, 210.0],
    [322.0, 208.0],
    [226.0, 209.0],
    [423.0, 200.0],
    [222.0, 200.0],
]
# Conversion factor (inches to meters)
METERS_PER_INCH = 0.0254

# Test points (not used to compute homography): [24, -12], [108, 12]
TEST_PTS_GROUND_PLANE = [[24, -12], [108, 12]]
TEST_PTS_IMAGE_PLANE = [[482.0, 287.0], [286.0, 202.5]]
HOMOGRAPHY_MATRIX = None

# Offset to transform left lens coordinates into base_link
CAMERA_TF_X = 10.5 * METERS_PER_INCH
CAMERA_TF_Y = 2.375 * METERS_PER_INCH


def get_homography_matrix():
    """Initialize data into a homography matrix."""
    # np_pts_ground = np.array(PTS_GROUND_PLANE)
    np_pts_ground = np.array([[56, -18.5], [56, 14], [23, -7], [31, 34.5]])
    np_pts_ground = np_pts_ground * METERS_PER_INCH
    np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

    # np_pts_image = np.array(PTS_IMAGE_PLANE)
    np_pts_image = np.array([[429, 202], [254, 202], [424, 256], [54, 226]])
    np_pts_image = np_pts_image * 1.0
    np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

    homography_matrix, err = cv2.findHomography(np_pts_image, np_pts_ground)

    # Find error of each point
    global HOMOGRAPHY_MATRIX
    HOMOGRAPHY_MATRIX = homography_matrix
    with open("errors.txt", "w") as fout:
        for image, ground in zip(PTS_IMAGE_PLANE + TEST_PTS_IMAGE_PLANE, PTS_GROUND_PLANE + TEST_PTS_GROUND_PLANE):
            x, y = transform_uv_to_xy(image[0], image[1])
            true_x = ground[0] * METERS_PER_INCH + CAMERA_TF_X
            true_y = ground[1] * METERS_PER_INCH + CAMERA_TF_Y
            x_error, y_error = x - true_x, y - true_y
            fout.write(f"{ground} = x error: {x_error:.3f} m, y error: {y_error:.3f} m\n")
    return homography_matrix


def transform_uv_to_xy(u, v):
    """
    u and v are pixel coordinates.
    The top left pixel is the origin, u axis increases to right, and v axis
    increases down.

    Returns a 1x2 numpy matrix of (x, y) displacement vector from base link
    to the point on the ground.
    Camera points along positive x axis and y axis increases to the left of
    the camera. Units are in meters.
    """
    homogeneous_point = np.array([[u], [v], [1]])
    xy = np.dot(HOMOGRAPHY_MATRIX, homogeneous_point)
    z = xy[2, 0]
    if z == 0:
        # Avoid error
        z = 1e-6
    x = xy[0, 0] / z
    y = xy[1, 0] / z
    # Camera offset
    x += CAMERA_TF_X
    y += CAMERA_TF_Y
    return x, y


def transform_xy_to_uv(x, y):
    """
    x and y are coordinates on the ground plane in meters.
    Returns the pixel coordinates (u, v) corresponding to that ground point.
    Top-left pixel is (0,0), u increases to the right, v increases downward.
    """
    # Camera offset
    x -= CAMERA_TF_X
    y -= CAMERA_TF_Y
    world_point = np.array([[x], [y], [1.0]])
    uvw = INV_HOMOGRAPHY_MATRIX.dot(world_point)
    w = uvw[2, 0]
    if w == 0:
        # Avoid error
        w = 1e-6
    u = uvw[0, 0] / w
    v = uvw[1, 0] / w
    return u, v


HOMOGRAPHY_MATRIX = get_homography_matrix()
INV_HOMOGRAPHY_MATRIX = np.linalg.inv(HOMOGRAPHY_MATRIX)
