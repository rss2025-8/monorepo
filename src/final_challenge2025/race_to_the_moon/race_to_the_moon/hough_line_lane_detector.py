#!/usr/bin/env python

import cv2
import numpy as np
import rclpy

# import your color segmentation algorithm; call this function in ros_image_callback!
# from computer_vision.color_segmentation import cd_color_segmentation, image_print
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point  # geometry_msgs not in CMake file
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from vs_msgs.msg import LinePixels, Trajectory
from race_to_the_moon.race_to_the_moon.computer_vision.lane_detection import get_lane_rho_thetas, rho_theta_to_pxs
from race_to_the_moon.race_to_the_moon.computer_vision.color_segmentation import cd_color_segmentation
from race_to_the_moon.visualization_tools import VisualizationTools
from race_to_the_moon.race_to_the_moon.computer_vision.homography_utils import HomographyTransformer
import math

class LaneDetector(Node):
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """

    def __init__(self):
        super().__init__("lane_detector")

        self.declare_parameter('hough_method', 'probabilistic')
        self.declare_parameter('left_lane_min_angle', -45)
        self.declare_parameter('left_lane_max_angle', -15)
        self.declare_parameter('right_lane_min_angle', 15)
        self.declare_parameter('right_lane_max_angle', 45)
        self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')

        self.hough_method = self.get_parameter('hough_method').value
        self.left_min = self.get_parameter('left_lane_min_angle').value
        self.left_max = self.get_parameter('left_lane_max_angle').value
        self.right_min = self.get_parameter('right_lane_min_angle').value
        self.right_max = self.get_parameter('right_lane_max_angle').value
        self.image_topic = self.get_parameter('image_topic').value

        self.lower_color = self.get_parameter('lower_color').value
        self.upper_color = self.get_parameter('upper_color').value

        self.left_lane_color = self.get_parameter('left_lane_color').value
        self.right_lane_color = self.get_parameter('right_lane_color').value
        self.midline_color = self.get_parameter('midline_color').value

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

        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 5)
        self.bridge = CvBridge()

        self.left_lane_pub = self.create_publisher(Marker, "/left_lane_marker", 10)
        self.right_lane_pub = self.create_publisher(Marker, "/right_lane_marker", 10)
        self.midline_visualization_pub = self.create_publisher(Marker, "/midline_marker", 10)
        self.midline_pub = self.create_publisher(Trajectory, "/trajectory", 10)

        # self.declare_parameter('hough_method', 'probabilistic')
        # self.declare_parameter('left_lane_min_angle', -45)
        # self.declare_parameter('left_lane_max_angle', -15)
        # self.declare_parameter('right_lane_min_angle', 15)
        # self.declare_parameter('right_lane_max_angle', 45)

        # self.hough_method = self.get_parameter('hough_method').value
        # self.left_min = self.get_parameter('left_lane_min_angle').value
        # self.left_max = self.get_parameter('left_lane_max_angle').value
        # self.right_min = self.get_parameter('right_lane_min_angle').value
        # self.right_max = self.get_parameter('right_lane_max_angle').value

        self.hough_line_params = (
            self.hough_method,
            (self.left_min, self.left_max),
            (self.right_min, self.right_max)
        )

        self.homography_transformer = HomographyTransformer(self.)
        (self, PTS_IMAGE_PLANE: str, PTS_GROUND_PLANE: str, METERS_PER_INCH: float):

    def image_callback(self, image_msg):
        # Apply your imported color segmentation function (cd_color_segmentation) to the image msg here
        # From your bounding box, take the center pixel on the bottom
        # (We know this pixel corresponds to a point on the ground plane)
        # publish this pixel (u, v) to the /relative_cone_px topic; the homography transformer will
        # convert it to the car frame.

        #################################
        # YOUR CODE HERE
        # detect the cone and publish its
        # pixel location in the image.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        #################################
        # img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        # size_x, _, _ = image.shape
        white_line_mask = cd_color_segmentation(image, self.lower_color, self.upper_color)
        left_line_rho_thetas, right_line_rho_thetas = get_lane_rho_thetas(self.hough_line_params, white_line_mask)

        if np.all(left_line_rho_thetas is not None) and np.all(right_line_rho_thetas is not None):

            midline_rho_thetas = 1/2 * (left_line_rho_thetas + right_line_rho_thetas)

            left_line_pxs = rho_theta_to_pxs(left_line_rho_thetas)
            right_line_pxs = rho_theta_to_pxs(right_line_rho_thetas)
            midline_pxs = rho_theta_to_pxs(midline_rho_thetas)

            left_lane_relxy = self.homography_transformer.transformLine(left_line_pxs)
            right_lane_relxy = self.homography_transformer.transformLine(right_line_pxs)
            midline_relxy = self.homography_transformer.transformLine(midline_pxs)

            VisualizationTools.plot_line(left_lane_relxy, color=self.left_lane_color, frame="base_link")
            VisualizationTools.plot_line(right_lane_relxy, color=self.right_lane_color, frame="base_link")
            VisualizationTools.plot_line(midline_relxy, color=self.midline_color, frame="base_link")

            midline = Trajectory()
            start_point = Point()
            end_point = Point()
            start_point.x = float(midline[0])
            start_point.y = float(midline[2])
            end_point.x = float(midline[1])
            end_point.y = float(midline[3])

            midline.points = [start_point, end_point]

            self.midline_pub.publish(midline)

        # These should be in np.ndarray([x0, y0, x1, y1]) format. xs increase left to right and ys increase downward
        # Add in steering functionality to steer it back if needed if one of the lines isn't there
        # if np.all(left_line_pxs is not None) and np.all(right_line_pxs is not None):
        #     midline = 1/2 * (left_line_pxs + right_line_pxs)
        #     lane_pixels = LanePixels()
        #     left_lane = LinePixels()
        #     right_lane = LinePixels()
        #     lane_midline = LinePixels()
        #     left_lane.u0 = float(left_line_pxs[0])
        #     left_lane.v0 = float(left_line_pxs[1])
        #     left_lane.u1 = float(left_line_pxs[2])
        #     left_lane.v1 = float(left_line_pxs[3])

        #     right_lane.u0 = float(right_line_pxs[0])
        #     right_lane.v0 = float(right_line_pxs[1])
        #     right_lane.u1 = float(right_line_pxs[2])
        #     right_lane.v1 = float(right_line_pxs[3])

        #     lane_midline.u0 = float(midline[0])
        #     lane_midline.v0 = float(midline[1])
        #     lane_midline.u1 = float(midline[2])
        #     lane_midline.v1 = float(midline[3])

        #     lane_pixels.left_lane = left_lane
        #     lane_pixels.right_lane = right_lane
        #     lane_pixels.lane_midline = lane_midline

        #     self.lane_pub.publish(lane_pixels)
        # if left_line_pxs is not None and right_line_pxs is None:
        #     right_line_pxs = np.copy(left_line_pxs)
        #     right_line_pxs[0] = float(size_x)
        #     right_line_pxs[2] = float(size_x)
        # elif left_line_pxs is None and right_line_pxs is not None:
        #     left_line_pxs = np.copy(right_line_pxs)
        #     left_line_pxs[0] = 0.
        #     left_line_pxs[2] = 0.
        # if left_line_pxs is not None:
        #     left_msg = LinePixels()
        #     left_msg.u0 = left_line_pxs[0]
        #     left_msg.v0 = left_line_pxs[1]
        #     left_msg.u1 = left_line_pxs[2]
        #     left_msg.v1 = left_line_pxs[3]
        #     self.left_lane_pub.publish(left_msg)
        # if right_line_pxs is not None:
        #     right_msg = LinePixels()
        #     right_msg.u0 = right_line_pxs[0]
        #     right_msg.v0 = right_line_pxs[1]
        #     right_msg.u1 = right_line_pxs[2]
        #     right_msg.v1 = right_line_pxs[3]
        #     self.right_lane_pub.publish(right_msg)

        # left line should be (x0, y0, x1, y1) and same with right line
        # if either is none, but not both, publish to steering forcing topic
        # if both are none, publish to safety controller topic??? - optional parameter


    def visualize_lane_points(self, msg, publisher, color):
        """Process lane points from pixel coordinates then visualize"""
        u0, v0 = msg.u0, msg.v0
        u1, v1 = msg.u1, msg.v1

        x0, y0 = self.transformUvToXy(u0, v0)
        x1, y1 = self.transformUvToXy(u1, v1)

        self.get_logger().info(f"Lane line detected at pixels: ({u0},{v0}) to ({u1},{v1})")
        self.get_logger().info(f"Transformed to ground coordinates: ({x0},{y0}) to ({x1},{y1})")

        VisualizationTools.plot_line([x0, x1], [y0, y1], publisher, color=color, frame="base_link")

def main(args=None):
    rclpy.init(args=args)
    lane_detector = LaneDetector()
    rclpy.spin(lane_detector)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
