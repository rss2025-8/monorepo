"""Takes in the camera image and publishes the location of the detected lanes."""

#!/usr/bin/env python

import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
from vs_msgs.msg import Point, Trajectory

from . import visualize
from .computer_vision import line_utils
from .computer_vision.color_segmentation import color_segmentation_white


class LaneDetector(Node):
    def __init__(self):
        super().__init__("lane_detector")

        self.hough_method: str = self.declare_parameter("hough_method", "probabilistic").value

        # Line angles are in [-90, 90] degrees, with 0 degrees being horizontal (-), 45 degrees (/), -45 degrees (\)
        self.left_min: float = self.declare_parameter("left_lane_min_angle", 1).value
        self.left_max: float = self.declare_parameter("left_lane_max_angle", 75).value
        self.right_min: float = self.declare_parameter("right_lane_min_angle", -75).value
        self.right_max: float = self.declare_parameter("right_lane_max_angle", -1).value

        self.image_topic: str = self.declare_parameter("image_topic", "/zed/zed_node/rgb/image_rect_color").value
        self.debug: bool = self.declare_parameter("debug", True).value
        self.disable_topic: str = self.declare_parameter("disable_topic", "/temp_disable").value  # Safety controller

        self.left_lane_color: tuple = self.declare_parameter("left_lane_color", (0.0, 1.0, 0.0)).value
        self.right_lane_color: tuple = self.declare_parameter("right_lane_color", (0.0, 1.0, 0.0)).value
        self.midline_color: tuple = self.declare_parameter("midline_color", (0.0, 0.0, 1.0)).value
        self.forced_midline_color: tuple = self.declare_parameter("forced_midline_color", (1.0, 0.0, 0.0)).value

        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 5)
        self.bridge = CvBridge()

        self.left_lane_pub = self.create_publisher(Marker, "/race/left_lane", 10)
        self.right_lane_pub = self.create_publisher(Marker, "/race/right_lane", 10)
        self.midline_pub = self.create_publisher(Marker, "/race/mid_lane", 10)
        self.trajectory_pub = self.create_publisher(Trajectory, "/race/trajectory", 10)
        self.debug_image_pub = self.create_publisher(Image, "/race/debug_img", 10)
        self.debug_image_pub_2 = self.create_publisher(Image, "/race/debug_img_2", 10)
        self.disable_pub = self.create_publisher(Bool, self.disable_topic, 10)

        # Debug params 5/8
        self.force_both_lanes = True
        self.image_width = 672.

        # Track runtime
        self.timing = [0, 0.0]

        self.hough_line_params = (self.hough_method, (self.left_min, self.left_max), (self.right_min, self.right_max))
        self.last_left_line = None
        self.last_right_line = None

        if self.debug:
            self.get_logger().info("DEBUG mode enabled")
        self.get_logger().info(f"Hough line params: {self.hough_line_params}")
        self.get_logger().info("Lane detector initialized")

    def image_callback(self, image_msg):
        call_time = self.get_clock().now()
        # Convert image to cv2 format
        original_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        # TODO Parameter mask out the top part of the image (replace with black)
        image = original_image.copy()
        portion_to_mask = 0.5
        image[0 : int(image.shape[0] * portion_to_mask), :] = 0

        # Find white lines in image
        white_line_mask = color_segmentation_white(image)
        # Get rho and theta for each line, in uv coordinates
        left_line, right_line = line_utils.get_best_lanes(
            self.hough_line_params, white_line_mask, self.last_left_line, self.last_right_line
        )

        # Plot lines that exist
        if left_line:
            left_line_uv = line_utils.rho_theta_to_endpoints(np.array(left_line))
            left_lane_xy = line_utils.endpoints_uv_to_xy(left_line_uv)
            # self.get_logger().info(f"Left line: {left_line_uv}")
            visualize.plot_line(
                left_lane_xy[:, 0],
                left_lane_xy[:, 1],
                self.left_lane_pub,
                color=self.left_lane_color,
                z=0.05,
                frame="base_link",
            )
        if right_line:
            right_line_uv = line_utils.rho_theta_to_endpoints(np.array(right_line))
            right_lane_xy = line_utils.endpoints_uv_to_xy(right_line_uv)
            # self.get_logger().info(f"Right line: {right_line_uv}")
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
            # Convert to car coordinates first
            left_line_uv = line_utils.rho_theta_to_endpoints(np.array(left_line))
            left_line_xy = line_utils.endpoints_uv_to_xy(left_line_uv)
            left_line_car = line_utils.endpoints_to_rho_theta(left_line_xy)

            right_line_uv = line_utils.rho_theta_to_endpoints(np.array(right_line))
            right_line_xy = line_utils.endpoints_uv_to_xy(right_line_uv)
            right_line_car = line_utils.endpoints_to_rho_theta(right_line_xy)

            mid_line_car = line_utils.find_midline_rho_theta(left_line_car, right_line_car, is_in_uv=False)
            # TODO find one closer to the left
            # mid_line_car = line_utils.find_midline_rho_theta(left_line_car, mid_line_car, is_in_uv=False)
            mid_line_xy = line_utils.rho_theta_to_endpoints(mid_line_car)
            # self.get_logger().info(f"Left line: {left_line_car}")
            # self.get_logger().info(f"Right line: {right_line_car}")
            # self.get_logger().info(f"Midline: {mid_line_car}")
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

            # Disable the safety controller (on track)
            disable_msg = Bool()
            disable_msg.data = True
            self.disable_pub.publish(disable_msg)
        elif self.force_both_lanes and (left_line or right_line):
            if not left_line:
                mid_line_xy_forced = line_utils.endpoints_uv_to_xy(np.array([[0, 5.0], [0, 0]]))
                mid_line_xy = mid_line_xy_forced
                visualize.plot_line(
                    mid_line_xy[:, 0],
                    mid_line_xy[:, 1],
                    self.midline_pub,
                    color=self.forced_midline_color,
                    z=0.05,
                    frame="base_link",
                )
            elif not right_line:
                mid_line_xy_forced = line_utils.endpoints_uv_to_xy(np.array([[self.image_width, 5.0], [self.image_width, 0]]))
                mid_line_xy = mid_line_xy_forced
                visualize.plot_line(
                    mid_line_xy[:, 0],
                    mid_line_xy[:, 1],
                    self.midline_pub,
                    color=self.forced_midline_color,
                    z=0.05,
                    frame="base_link",
                )
            else:
                raise Exception

        if self.debug:
            # Draw lines on image
            # edges = line_utils.find_edges(white_line_mask)
            # debug_msg = self.bridge.cv2_to_imgmsg(edges, "mono8")
            # self.debug_image_pub_2.publish(debug_msg)

            masked_image = cv2.bitwise_and(image, image, mask=white_line_mask)
            lines_image = masked_image.copy()

            def plot_line(rho, theta, color=(0.0, 1.0, 0.0), thickness=2):
                # RGB to BGR
                color = (color[2], color[1], color[0])
                color = tuple(int(c * 255) for c in color)
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(lines_image, pt1, pt2, color, thickness, cv2.LINE_AA)

            left_lines, right_lines = line_utils.get_all_lanes(self.hough_line_params, white_line_mask)
            left_lines, right_lines = line_utils.filter_close_lanes(
                left_lines, right_lines, self.last_left_line, self.last_right_line
            )
            raw_lines = line_utils.get_raw_lines(self.hough_line_params, white_line_mask)
            # self.get_logger().info(f"Left lines: {left_lines}")
            # self.get_logger().info(f"Right lines: {right_lines}")

            # if self.last_left_line:
            #     plot_line(self.last_left_line[0], self.last_left_line[1], color=(1.0, 1.0, 0.0), thickness=5)
            # if self.last_right_line:
            #     plot_line(self.last_right_line[0], self.last_right_line[1], color=(1.0, 1.0, 0.0), thickness=5)
            for rho, theta in raw_lines:
                # duplicate = False
                # for left_rho, left_theta in left_lines:
                #     if abs(rho - left_rho) < 0.01 and abs(theta - left_theta) < 0.001:
                #         duplicate = True
                #         break
                # for right_rho, right_theta in right_lines:
                #     if abs(rho - right_rho) < 0.01 and abs(theta - right_theta) < 0.001:
                #         duplicate = True
                #         break
                # if not duplicate:
                plot_line(rho, theta, color=(0.5, 0.5, 0.5), thickness=1)
            # if left_line:
            #     plot_line(left_line[0], left_line[1], color=(0.0, 1.0, 0.0), thickness=5)
            # if right_line:
            #     plot_line(right_line[0], right_line[1], color=(0.0, 1.0, 0.0), thickness=5)
            # lines_image = cv2.addWeighted(original_image, 0.5, lines_image, 1, 0)
            # debug_msg = self.bridge.cv2_to_imgmsg(lines_image, "bgr8")
            # self.debug_image_pub_2.publish(debug_msg)
            # lines_image = masked_image.copy()
            # lines_image = np.zeros_like(image)
            for rho, theta in left_lines:
                plot_line(rho, theta, color=self.left_lane_color, thickness=2)
            for rho, theta in right_lines:
                plot_line(rho, theta, color=self.right_lane_color, thickness=2)

            plot_line(rho, theta, color=(1., 0., 0.), thickness=1)

            # left_line, right_line = line_utils.get_best_lanes(self.hough_line_params, white_line_mask)
            # if left_line and right_line:
            #     mid_line = line_utils.find_midline_rho_theta(left_line, right_line, is_in_uv=True)
            #     plot_line(mid_line[0], mid_line[1], color=self.midline_color)
            lines_image = cv2.addWeighted(original_image, 0.5, lines_image, 1, 0)

            debug_msg = self.bridge.cv2_to_imgmsg(lines_image, "bgr8")
            self.debug_image_pub.publish(debug_msg)

        # Update last lines
        self.last_left_line = left_line
        self.last_right_line = right_line
        # self.get_logger().info(f"Left line: {left_line}")
        # self.get_logger().info(f"Right line: {right_line}")

        latency = (self.get_clock().now() - call_time).nanoseconds / 1e9
        self.timing[0] += 1
        self.timing[1] += latency
        if self.timing[0] == 50:
            avg_latency = self.timing[1] / 50
            if avg_latency > 0.02:
                self.get_logger().warning(f"high lane detector latency, optimize: {avg_latency:.4f}s")
            else:
                self.get_logger().info(f"lanes: {avg_latency:.4f}s")
            self.timing = [0, 0.0]


def main(args=None):
    rclpy.init(args=args)
    lane_detector = LaneDetector()
    rclpy.spin(lane_detector)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
