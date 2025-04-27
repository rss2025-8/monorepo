#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import String
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from vs_msgs.msg import LanePixels, LookaheadLocation
from race_to_the_moon.visualization_tools import VisualizationTools
import math

class LaneHomographyTransformer(Node):
    def __init__(self):
        super().__init__("lane_homography_transformer")

        self.left_lane_pub = self.create_publisher(Marker, "/left_lane_marker", 10)
        self.right_lane_pub = self.create_publisher(Marker, "/right_lane_marker", 10)
        self.lane_midline_visualization_pub = self.create_publisher(Marker, "/lane_midline_marker", 10)
        self.lookahead_pub = self.create_publisher(LookaheadLocation, "/relative_lookahead", 10)

        self.left_lane_sub = self.create_subscription(LanePixels, "/lane_pxs", self.lane_detection_callback, 1)

        self.declare_parameter('lookahead_dist', 4)

        self.declare_parameter('use_homography', 'rss2025_8_old')
        homography_name = self.get_parameter('use_homography').value

        self.declare_parameter(f'HOMOGRAPHIES.{homography_name}.PTS_IMAGE_PLANE', "[ \
            [429, 202], \
            [254, 202], \
            [424, 256], \
            [54, 226] \
        ]")
        self.declare_parameter(f'HOMOGRAPHIES.{homography_name}.PTS_GROUND_PLANE', "[ \
            [56, -18.5], \
            [56, 14], \
            [23, -7], \
            [31, 34.5] \
        ]")

        self.PTS_IMAGE_PLANE = eval(self.get_parameter(f'HOMOGRAPHIES.{homography_name}.PTS_IMAGE_PLANE').value)
        self.PTS_GROUND_PLANE = eval(self.get_parameter(f'HOMOGRAPHIES.{homography_name}.PTS_GROUND_PLANE').value)

        self.declare_parameter('left_lane_color', (0.0, 1.0, 0.0))
        self.declare_parameter('right_lane_color', (1.0, 0.0, 0.0))
        self.declare_parameter('lane_midline_color', (0.0, 0.0, 1.0))

        self.declare_parameter('meters_per_inch', 0.0254)

        # self.PTS_IMAGE_PLANE = self.get_parameter('pts_image_plane').value
        # self.PTS_GROUND_PLANE = self.get_parameter('pts_ground_plane').value
        self.METERS_PER_INCH = self.get_parameter('meters_per_inch').value

        self.left_lane_color = self.get_parameter('left_lane_color').value
        self.right_lane_color = self.get_parameter('right_lane_color').value
        self.lane_midline_color = self.get_parameter('lane_midline_color').value

        self.lookahead_dist = self.get_parameter('lookahead_dist').value

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

    def lane_detection_callback(self, msg):
        left_lane = msg.left_lane
        right_lane = msg.right_lane
        lane_midline = msg.lane_midline
        self.visualize_lane_points(left_lane, self.left_lane_pub, self.left_lane_color)  # Green
        self.visualize_lane_points(right_lane, self.right_lane_pub, self.right_lane_color) # Red
        self.visualize_lane_points(lane_midline, self.lane_midline_visualization_pub, self.lane_midline_color)
        self.publish_lookahead_point(lane_midline)


    def visualize_lane_points(self, msg, publisher, color):
        """Process lane points from pixel coordinates then visualize"""
        u0, v0 = msg.u0, msg.v0
        u1, v1 = msg.u1, msg.v1

        x0, y0 = self.transformUvToXy(u0, v0)
        x1, y1 = self.transformUvToXy(u1, v1)

        self.get_logger().info(f"Lane line detected at pixels: ({u0},{v0}) to ({u1},{v1})")
        self.get_logger().info(f"Transformed to ground coordinates: ({x0},{y0}) to ({x1},{y1})")

        VisualizationTools.plot_line([x0, x1], [y0, y1], publisher, color=color, frame="base_link")

    def publish_lookahead_point(self, msg):
        """Process lane points from pixel coordinates then visualize"""
        u0, v0 = msg.u0, msg.v0
        u1, v1 = msg.u1, msg.v1

        # (u,v) = f(m) = mV + B Any continuous point along the midpoint line can be represented as a scaled vector + initial point b
        # B = np.array([[u0], [v0]])
        # unnormalized_V = np.array([[u1-u0], [v1-v0]])
        # V = unnormalized_V / np.linalg.norm(unnormalized_V)

        x0, y0 = self.transformUvToXy(u0, v0)
        x1, y1 = self.transformUvToXy(u1, v1)

        s1 = np.array([x0,y0])
        s2 = np.array([x1,y1])

        lookahead_point = self.get_lookahead_point(s1, s2)

        if lookahead_point is not None:
            lookahead_msg = LookaheadLocation()
            lookahead_msg.x_pos = float(lookahead_point[0])
            lookahead_msg.y_pos = float(lookahead_point[1])
            self.lookahead_pub.publish(lookahead_msg)

    def get_lookahead_point(self, s1, s2):
        """Returns the next goal point (x, y), which is a set lookahead dist from the car, or None if no point is found.
        Also returns the index of the segment the lookahead point is on.

        car_pose is (x, y, theta).
        nearest_segment_idx = i is the first segment (points[i], points[i+1]) that's not behind the car.
        Assumes points[i+1] is further along the path than points[i].
        """
        # Find the first segment ahead of the car that intersects the circle
        intersections = self.circle_segment_intersections(s1, s2)
        valid_intersections = []
        car_forward_vec = np.array([1, 0])

        for p in intersections:
            if np.dot(car_forward_vec, p - s1) >= 0:  # Ensure the point is ahead of the car
                valid_intersections.append(p)

        if valid_intersections:
            valid_intersections.sort(key=lambda p: np.linalg.norm(p - s2))
            return valid_intersections[0]
        return None

    def circle_segment_intersections(self, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        """Returns intersection points (0, 1, or 2, (x, y)) of a circle (center c and radius r) and a line segment."""
        c = np.array([0,0,])
        r = self.lookahead_dist
        # Special case: Segment is a single point
        if np.linalg.norm(s1 - s2) < 1e-6:
            if np.isclose(np.linalg.norm(s1 - c), r):
                return np.array([s1])  # Point on circle
            else:
                return np.empty((0, 2), dtype=float)  # Point not on circle

        # Solve at^2 + bt + c = 0, where t sweeps along the segment
        # Point = s1 + t * v (0 <= t <= 1)
        v = s2 - s1  # Vector along line segment
        a = np.dot(v, v)
        b = 2 * np.dot(v, s1 - c)
        c = np.dot(s1, s1) + np.dot(c, c) - 2 * np.dot(s1, c) - r**2

        disc = b**2 - 4 * a * c  # Discriminant
        if disc < -1e-6:  # No intersection
            return np.empty((0, 2), dtype=float)
        elif disc < 0:
            disc = 0  # Numerical tolerance

        intersections = []
        # Up to two possible intersections
        sqrt_disc = np.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)
        if -1e-6 <= t1 <= 1 + 1e-6:
            intersections.append(s1 + t1 * v)  # First point
        if -1e-6 <= t2 <= 1 + 1e-6 and abs(t2 - t1) > 1e-6:
            intersections.append(s1 + t2 * v)  # Second point that's not a duplicate

        if intersections:
            return np.array(intersections)
        else:
            return np.empty((0, 2), dtype=float)

    def transformUvToXy(self, u, v):
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
        scaling_factor = 1.0 / xy[2, 0]
        homogeneous_xy = xy * scaling_factor
        x = homogeneous_xy[0, 0]
        y = homogeneous_xy[1, 0]
        return x, y

def main(args=None):
    rclpy.init(args=args)
    lane_homography_transformer = LaneHomographyTransformer()
    rclpy.spin(lane_homography_transformer)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
