import numpy as np
import cv2


class HomographyTransformer():
    def __init__(self, PTS_IMAGE_PLANE: str, PTS_GROUND_PLANE: str, METERS_PER_INCH: float):
        self.PTS_IMAGE_PLANE = eval(PTS_IMAGE_PLANE)
        self.PTS_GROUND_PLANE = eval(PTS_GROUND_PLANE)
        self.METERS_PER_INCH = METERS_PER_INCH

        if not len(self.PTS_GROUND_PLANE) == len(self.PTS_IMAGE_PLANE):
            self.get_logger().error("ERROR: PTS_GROUND_PLANE and PTS_IMAGE_PLANE should be of same length")
            return

        self.update_homography_matrix()


    def update_homography_matrix(self):
        """Updates the homography matrix based on current parameters"""
        np_pts_ground = np.array(self.PTS_GROUND_PLANE)
        np_pts_ground = np_pts_ground * self.METERS_PER_INCH
        np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

        np_pts_image = np.array(self.PTS_IMAGE_PLANE)
        np_pts_image = np_pts_image * 1.0
        np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

        self.h, err = cv2.findHomography(np_pts_image, np_pts_ground)

    # def transformUvToXy(self, u, v):
    #     """
    #     u and v are pixel coordinates.
    #     The top left pixel is the origin, u axis increases to right, and v axis
    #     increases down.

    #     Returns a normal non-np 1x2 matrix of xy displacement vector from the
    #     camera to the point on the ground plane.
    #     Camera points along positive x axis and y axis increases to the left of
    #     the camera.

    #     Units are in meters.
    #     """
    #     homogeneous_point = np.array([[u], [v], [1]])
    #     xy = np.dot(self.h, homogeneous_point)
    #     scaling_factor = 1.0 / xy[2, 0]
    #     homogeneous_xy = xy * scaling_factor
    #     x = homogeneous_xy[0, 0]
    #     y = homogeneous_xy[1, 0]
    #     return x, y

    def transform_uv_to_xy(self, u, v):
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
        xy = np.dot(self.h, homogeneous_point)
        z = xy[2, 0]
        if z == 0:
            # avoid error
            z = 1e-6
        x = xy[0, 0] / z
        y = xy[1, 0] / z
        return x, y

    # def transformLine(self, line: np.ndarray):
    #     # u0, v0 = line[0], line[1]
    #     # u1, v1 = line[2], line[3]
    #     u0, v0 = line[0, 0], line[0, 1]
    #     u1, v1 = line[0, 2], line[0, 3]

    #     x0, y0 = self.transformUvToXy(u0, v0)
    #     x1, y1 = self.transformUvToXy(u1, v1)

    #     return np.array([[x0, y0], [x1, y1]])

    def transformLine(self, line: np.ndarray):
        # u0, v0 = line[0], line[1]
        # u1, v1 = line[2], line[3]
        u0, v0 = line[0, 0], line[0, 1]
        u1, v1 = line[0, 2], line[0, 3]

        x0, y0 = self.transform_uv_to_xy(u0, v0)
        x1, y1 = self.transform_uv_to_xy(u1, v1)

        return np.array([[x0, y0], [x1, y1]])
