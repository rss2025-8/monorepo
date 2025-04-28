import cv2
import numpy as np

PTS_IMAGE_PLANE = [[429, 202], [254, 202], [424, 256], [54, 226]]
PTS_GROUND_PLANE = [[56, -18.5], [56, 14], [23, -7], [31, 34.5]]
METERS_PER_INCH = 0.0254


def get_homography_matrix():
    """Initialize data into a homography matrix."""
    np_pts_ground = np.array(PTS_GROUND_PLANE)
    np_pts_ground = np_pts_ground * METERS_PER_INCH
    np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

    np_pts_image = np.array(PTS_IMAGE_PLANE)
    np_pts_image = np_pts_image * 1.0
    np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

    homography_matrix, err = cv2.findHomography(np_pts_image, np_pts_ground)
    return homography_matrix


HOMOGRAPHY_MATRIX = get_homography_matrix()
INV_HOMOGRAPHY_MATRIX = np.linalg.inv(HOMOGRAPHY_MATRIX)


def transform_uv_to_xy(u, v):
    """
    u and v are pixel coordinates.
    The top left pixel is the origin, u axis increases to right, and v axis
    increases down.

    Returns a normal non-np 1x2 matrix of xy displacement vector from the
    camera to the point on the ground plane.
    Camera points along positive x axis and y axis increases to the left of
    the camera.

    Units are in meters.
    """
    homogeneous_point = np.array([[u], [v], [1]])
    xy = np.dot(HOMOGRAPHY_MATRIX, homogeneous_point)
    z = xy[2, 0]
    if z == 0:
        # avoid error
        z = 1e-6
    x = xy[0, 0] / z
    y = xy[1, 0] / z
    return x, y


def transform_xy_to_uv(x, y):
    """
    x and y are coordinates on the ground plane in meters.
    Returns the pixel coordinates (u, v) corresponding to that ground point.
    Top-left pixel is (0,0), u increases to the right, v increases downward.
    """
    world_point = np.array([[x], [y], [1.0]])
    uvw = INV_HOMOGRAPHY_MATRIX.dot(world_point)
    w = uvw[2, 0]
    if w == 0:
        # avoid error
        w = 1e-6
    u = uvw[0, 0] / w
    v = uvw[1, 0] / w
    return u, v
