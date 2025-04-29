"""Takes in the camera image and publishes the location of the detected lanes."""

#!/usr/bin/env python

import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from vs_msgs.msg import Point, Trajectory

from . import homography, visualize
from .computer_vision.color_segmentation import color_segmentation_white
from .computer_vision.line_detection import (
    circular_mean,
    get_all_lanes,
    get_best_lanes,
    get_raw_lines,
    rho_theta_to_pxs,
)


def find_mid_line(left_line, right_line):
    """Given two lines (rho_i, theta_i), return the midline (rho, theta) with theta in [-pi/2, pi/2].

    Lines are defined by rho = x * cos(theta) + y * sin(theta).
    This function finds the line that crosses the intersection point of the two lines,
    and has an averaged direction of the two lines.
    Assumes the lines are not parallel!
    """
    rho0, theta0 = left_line
    rho1, theta1 = right_line
    # Find intersection point of the two lines
    A = np.array([[np.cos(theta0), np.sin(theta0)], [np.cos(theta1), np.sin(theta1)]])
    b = np.array([rho0, rho1])
    assert abs(np.linalg.det(A)) >= 1e-8
    x_int, y_int = np.linalg.solve(A, b)

    # Find similar unit direction vectors along each line, ensuring they both point upwards
    d0 = np.array([-np.sin(theta0), np.cos(theta0)])
    if d0[1] > 0:
        d0 = -d0
    d1 = np.array([-np.sin(theta1), np.cos(theta1)])
    if d1[1] > 0:
        d1 = -d1

    # Find the average direction
    d = d0 + d1
    assert np.linalg.norm(d) > 1e-8
    d /= np.linalg.norm(d)  # [-sin(theta), cos(theta)]

    # Compute rho, theta for the midline
    theta = np.arcsin(-d[0])
    rho = x_int * np.cos(theta) + y_int * np.sin(theta)
    return rho, theta


class LaneDetector(Node):
    def __init__(self):
        super().__init__("lane_detector")

        self.declare_parameter("hough_method", "probabilistic")
        # Line angles are in [-90, 90] degrees, with 0 degrees being horizontal (-), 45 degrees (/), -45 degrees (\)
        self.declare_parameter("left_lane_min_angle", 45)
        self.declare_parameter("left_lane_max_angle", 75)
        self.declare_parameter("right_lane_min_angle", -75)
        self.declare_parameter("right_lane_max_angle", -45)
        self.declare_parameter("image_topic", "/zed/zed_node/rgb/image_rect_color")

        self.hough_method = self.get_parameter("hough_method").value
        self.left_min = self.get_parameter("left_lane_min_angle").value
        self.left_max = self.get_parameter("left_lane_max_angle").value
        self.right_min = self.get_parameter("right_lane_min_angle").value
        self.right_max = self.get_parameter("right_lane_max_angle").value
        self.image_topic = self.get_parameter("image_topic").value
        self.debug: bool = self.declare_parameter("debug", True).value

        self.declare_parameter("left_lane_color", (0.0, 1.0, 0.0))
        self.declare_parameter("right_lane_color", (1.0, 0.0, 0.0))
        self.declare_parameter("midline_color", (0.0, 0.0, 1.0))

        self.left_lane_color = self.get_parameter("left_lane_color").value
        self.right_lane_color = self.get_parameter("right_lane_color").value
        self.midline_color = self.get_parameter("midline_color").value

        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 5)
        self.bridge = CvBridge()

        self.left_lane_pub = self.create_publisher(Marker, "/race/left_lane", 10)
        self.right_lane_pub = self.create_publisher(Marker, "/race/right_lane", 10)
        self.midline_pub = self.create_publisher(Marker, "/race/mid_lane", 10)
        self.trajectory_pub = self.create_publisher(Trajectory, "/race/trajectory", 10)
        self.debug_image_pub = self.create_publisher(Image, "/race/debug_img", 10)

        self.hough_line_params = (self.hough_method, (self.left_min, self.left_max), (self.right_min, self.right_max))

        if self.debug:
            self.get_logger().info("DEBUG mode enabled")
        self.get_logger().info(f"Hough line params: {self.hough_line_params}")
        self.get_logger().info("Lane detector initialized")

    def transform_line_uv_to_xy(self, line_uv, amount_in_front=5.0):
        """
        Transforms a line from uv to xy. Returns a 2x2 numpy array with A[0] = (x0, y0), A[1] = (x1, y1).
        Guarantees that x0 = 0 and x1 = amount_in_front.
        """
        u0, v0 = line_uv[0], line_uv[1]
        u1, v1 = line_uv[2], line_uv[3]
        x0, y0 = homography.transform_uv_to_xy(u0, v0)
        x1, y1 = homography.transform_uv_to_xy(u1, v1)
        # Ensure x0 < x1
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        # Calculate point on line that has x = 0
        dx = x1 - x0
        if abs(dx) < 1e-6:
            dx = 1e-6  # Avoid division by zero
        t = -x0 / dx
        new_x0, new_y0 = 0.0, y0 + t * (y1 - y0)
        # Calculate point on line that has x = amount_in_front
        t = (amount_in_front - x0) / dx
        new_x1, new_y1 = amount_in_front, y0 + t * (y1 - y0)
        return np.array([[new_x0, new_y0], [new_x1, new_y1]])

    def image_callback(self, image_msg):
        # Convert image to cv2 format
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        # Find white lines in image
        white_line_mask = color_segmentation_white(image)
        # Get rho and theta for each line
        left_line, right_line = get_best_lanes(self.hough_line_params, white_line_mask)

        # Plot lines that exist
        if left_line:
            left_line_pxs = rho_theta_to_pxs(np.array(left_line))
            left_lane_xy = self.transform_line_uv_to_xy(left_line_pxs)
            # self.get_logger().info(f"Left line: {left_line_pxs}")
            visualize.plot_line(
                left_lane_xy[:, 0],
                left_lane_xy[:, 1],
                self.left_lane_pub,
                color=self.left_lane_color,
                z=0.05,
                frame="base_link",
            )
        if right_line:
            right_line_pxs = rho_theta_to_pxs(np.array(right_line))
            right_lane_xy = self.transform_line_uv_to_xy(right_line_pxs)
            # self.get_logger().info(f"Right line: {right_line_pxs}")
            visualize.plot_line(
                right_lane_xy[:, 0],
                right_lane_xy[:, 1],
                self.right_lane_pub,
                color=self.right_lane_color,
                z=0.05,
                frame="base_link",
            )
        # If both lines exist, plot midline
        if left_line and right_line:
            mid_line = find_mid_line(left_line, right_line)
            # self.get_logger().info(f"Left line: {left_line}")
            # self.get_logger().info(f"Right line: {right_line}")
            # self.get_logger().info(f"Midline: {mid_line}")
            mid_line_pxs = rho_theta_to_pxs(mid_line)
            mid_line_xy = self.transform_line_uv_to_xy(mid_line_pxs)
            visualize.plot_line(
                mid_line_xy[:, 0],
                mid_line_xy[:, 1],
                self.midline_pub,
                color=self.midline_color,
                z=0.05,
                frame="base_link",
            )

            # Create and publish trajectory
            trajectory = Trajectory()
            start_point = Point()
            start_point.x = float(mid_line_xy[0, 0])
            start_point.y = float(mid_line_xy[0, 1])
            end_point = Point()
            end_point.x = float(mid_line_xy[1, 0])
            end_point.y = float(mid_line_xy[1, 1])
            trajectory.points = [start_point, end_point]
            self.trajectory_pub.publish(trajectory)

        if self.debug:
            # Draw lines on image
            lines_image = image.copy()

            def plot_line(rho, theta, color=(0.0, 1.0, 0.0)):
                # RGB to BGR
                color = (color[2], color[1], color[0])
                color = tuple(int(c * 255) for c in color)
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(lines_image, pt1, pt2, color, 2, cv2.LINE_AA)

            left_lines, right_lines = get_all_lanes(self.hough_line_params, white_line_mask)
            raw_lines = get_raw_lines(self.hough_line_params, white_line_mask)
            # self.get_logger().info(f"Left lines: {left_lines}")
            # self.get_logger().info(f"Right lines: {right_lines}")
            for rho, theta in left_lines:
                plot_line(rho, theta, color=self.left_lane_color)
            for rho, theta in right_lines:
                plot_line(rho, theta, color=self.right_lane_color)

            left_line, right_line = get_best_lanes(self.hough_line_params, white_line_mask)
            if left_line and right_line:
                mid_line = find_mid_line(left_line, right_line)
                plot_line(mid_line[0], mid_line[1], color=self.midline_color)

            # plot_line(200, 0, color=(0.5, 0.5, 0.5))  # 0 is vertical
            # plot_line(200, np.pi / 2, color=(1.0, 0.0, 0.0))  # np.pi / 2 is horizontal

            debug_msg = self.bridge.cv2_to_imgmsg(lines_image, "bgr8")
            # filtered_image = cv2.bitwise_and(image, image, mask=white_line_mask)
            # debug_msg = self.bridge.cv2_to_imgmsg(filtered_image, "bgr8")
            self.debug_image_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    lane_detector = LaneDetector()
    rclpy.spin(lane_detector)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
