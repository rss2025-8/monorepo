#!/usr/bin/env python

import cv2 as cv
import numpy as np
import rclpy

from cv_bridge import CvBridge, CvBridgeError
# from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from vs_msgs.msg import Point, Trajectory
from race_to_the_moon.computer_vision.hough_lines_utils import cluster_and_merge_lines, get_left_and_right_lane_candidates, filter_line_candidates_by_row_extent, select_best_line_candidate, filter_line_candidates_by_column_extent, select_best_line_candidate_by_midpoint_radius
from race_to_the_moon.computer_vision.color_segmentation import bgr_color_segmentation
from race_to_the_moon.visualization_tools import VisualizationTools
from race_to_the_moon.computer_vision.homography_utils import HomographyTransformer
from . import visualize
import math

class LaneDetector(Node):
    """
    A class for lane detection algorithms
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /trajectory (Trajectory).
    """

    def __init__(self):
        super().__init__("lane_detector")

        # self.declare_parameter('hough_method', 'probabilistic')
        # self.declare_parameter('left_lane_min_angle', -45)
        # self.declare_parameter('left_lane_max_angle', -15)
        # self.declare_parameter('right_lane_min_angle', 15)
        # self.declare_parameter('right_lane_max_angle', 45)
        self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')

        # Additional parameter declarations
        self.declare_parameter('lower_color', [0, 0, 190])
        self.declare_parameter('upper_color', [179, 25, 255])
        self.declare_parameter('left_lane_color', [0.0, 1.0, 0.0])
        self.declare_parameter('right_lane_color', [1.0, 0.0, 0.0])
        self.declare_parameter('midline_color', [0.0, 0.0, 1.0])
        self.declare_parameter('lookahead_dist', 4.0)
        self.declare_parameter('use_homography', 'rss2025_8_old')
        self.declare_parameter('meters_per_inch', 0.0254)

        # Hough line parameters declaration
        self.declare_parameter('rho_resolution', 5)
        self.declare_parameter('theta_resolution', np.pi / 30)
        self.declare_parameter('threshold', 50)
        self.declare_parameter('minLineLength', 20)
        self.declare_parameter('maxLineGap', 100)
        self.declare_parameter('max_saturation', 40.0)
        self.declare_parameter('min_value', 140.0)
        self.declare_parameter('min_angle_left', np.pi/12)
        self.declare_parameter('max_angle_left', np.pi/2)
        self.declare_parameter('min_angle_right', 0.)
        self.declare_parameter('max_angle_right', 11*np.pi/12)
        self.declare_parameter('row_mask_lower_threshold', 188) # Need to figure out the size of the image
        self.declare_parameter('image_width', 672)
        self.declare_parameter('image_height', 376)
        self.declare_parameter("x_intercept_min_samples", 1)
        self.declare_parameter("x_intercept_cluster_eps", 30)

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
        # self.declare_parameter(f'HOMOGRAPHIES.{homography_name}.PTS_IMAGE_PLANE', "default")
        # self.declare_parameter(f'HOMOGRAPHIES.{homography_name}.PTS_GROUND_PLANE', "default")
        self.PTS_IMAGE_PLANE = self.get_parameter(f'HOMOGRAPHIES.{homography_name}.PTS_IMAGE_PLANE').value
        self.PTS_GROUND_PLANE = self.get_parameter(f'HOMOGRAPHIES.{homography_name}.PTS_GROUND_PLANE').value
        self.METERS_PER_INCH = self.get_parameter('meters_per_inch').value

        # Get parameter values
        # self.hough_method = self.get_parameter('hough_method').value
        # self.left_min = self.get_parameter('left_lane_min_angle').value
        # self.left_max = self.get_parameter('left_lane_max_angle').value
        # self.right_min = self.get_parameter('right_lane_min_angle').value
        # self.right_max = self.get_parameter('right_lane_max_angle').value
        self.image_topic = self.get_parameter('image_topic').value
        # self.safety_topic = self.get_parameter('safety_topic').value

        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value

        self.lower_color = self.get_parameter('lower_color').value
        self.upper_color = self.get_parameter('upper_color').value
        self.left_lane_color = self.get_parameter('left_lane_color').value
        self.right_lane_color = self.get_parameter('right_lane_color').value
        self.midline_color = self.get_parameter('midline_color').value
        self.lookahead_dist = self.get_parameter('lookahead_dist').value

        self.x_intercept_cluster_eps = self.get_parameter('x_intercept_cluster_eps').value
        self.x_intercept_min_samples = self.get_parameter('x_intercept_min_samples').value

        # Create subscriptions and publishers
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 5)
        self.bridge = CvBridge()

        self.left_lane_pub = self.create_publisher(Marker, "/race/left_line", 10)
        self.right_lane_pub = self.create_publisher(Marker, "/race/right_line", 10)
        self.midline_visualization_pub = self.create_publisher(Marker, "/race/mid_line", 10)
        self.trajectory_pub = self.create_publisher(Trajectory, "/trajectory", 10)
        self.debug: bool = self.declare_parameter("debug", True).value
        self.debug_image_pub = self.create_publisher(Image, "/race/debug_img", 10)

        # Configure Hough line detection parameters
        self.hough_line_params = {
            "rho_resolution": self.get_parameter('rho_resolution').value,
            "theta_resolution": self.get_parameter('theta_resolution').value,
            "threshold": self.get_parameter('threshold').value,
            "minLineLength": self.get_parameter('minLineLength').value,
            "maxLineGap": self.get_parameter('maxLineGap').value,
            "max_saturation": self.get_parameter('max_saturation').value,
            "min_value": self.get_parameter('min_value').value,
            "min_angle_left": self.get_parameter('min_angle_left').value,
            "max_angle_left": self.get_parameter('max_angle_left').value,
            "min_angle_right": self.get_parameter('min_angle_right').value,
            "max_angle_right": self.get_parameter('max_angle_right').value,
            "row_mask_lower_threshold": self.get_parameter('row_mask_lower_threshold').value, # Better name for this?
            "x_intercept_cluster_eps": self.get_parameter("x_intercept_cluster_eps").value,  # Clustering epsilon parameter
            "x_intercept_min_samples": self.get_parameter("x_intercept_min_samples").value, # Min samples for clustering
        }

        # Initialize homography transformer
        self.homography_transformer = HomographyTransformer(
            self.PTS_IMAGE_PLANE,
            self.PTS_GROUND_PLANE,
            self.METERS_PER_INCH
        )

        if self.debug:
            self.get_logger().info("DEBUG mode enabled")
        self.get_logger().info(f"Hough line params: {self.hough_line_params}")
        self.get_logger().info("Lane detector initialized")

    def image_callback(self, image_msg):
        bgr_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        if bgr_image is None:
            return None

        white_lines = bgr_color_segmentation(bgr_image, self.hough_line_params["min_value"], self.hough_line_params["max_saturation"])

        canny = cv.Canny(white_lines, 50, 200, None, 3)

        bgr_canny = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)

        left_line_candidates, right_line_candidates = get_left_and_right_lane_candidates(
            canny, self.hough_line_params,
            (self.hough_line_params["min_angle_left"], self.hough_line_params["max_angle_left"]),
            (self.hough_line_params["min_angle_right"], self.hough_line_params["max_angle_right"])
        )

        filtered_left_lines = filter_line_candidates_by_row_extent(left_line_candidates, self.hough_line_params)
        filtered_right_lines = filter_line_candidates_by_row_extent(right_line_candidates, self.hough_line_params)

        merged_left_lines = cluster_and_merge_lines(
            filtered_left_lines,
            eps=self.hough_line_params.get("x_intercept_cluster_eps", 20),
            min_samples=self.hough_line_params.get("x_intercept_min_samples", 1)
        )
        merged_right_lines = cluster_and_merge_lines(
            filtered_right_lines,
            eps=self.hough_line_params.get("x_intercept_cluster_eps", 20),
            min_samples=self.hough_line_params.get("x_intercept_min_samples", 1)
        )

        column_filtered_left_lines = filter_line_candidates_by_column_extent(merged_left_lines, self.image_width, is_left=True)
        column_filtered_right_lines = filter_line_candidates_by_column_extent(merged_right_lines, self.image_width, is_left=False)

        best_left_line = self.choose_line(column_filtered_left_lines)
        best_right_line = self.choose_line(column_filtered_right_lines)

        # These are N by 7 ndarrays x1, y1, x2, y2, rho, theta, angle = line

        # Handle lane line detection scenarios
        if best_left_line is not None and best_right_line is not None:
            self.pub_visualization_and_trajectory(best_left_line, best_right_line)
        elif best_left_line is not None:
            # No right line detected - create virtual right line
            right_boundary_line = np.array([[self.image_width, 0, self.image_width, self.image_height, self.image_height/2, 0, np.pi/2]])
            self.pub_visualization_and_trajectory(best_left_line, right_boundary_line)
        elif best_right_line is not None:
            # No left line detected - create virtual left line
            # height, width = bgr_image.shape[:2]  # Fixed: use shape instead of size
            left_boundary_line = np.array([[0, 0, 0, self.image_height, self.image_height/2, 0, np.pi/2]])
            self.pub_visualization_and_trajectory(left_boundary_line, best_right_line)
        else:
            self.get_logger().warning("No lane lines detected")

        # if self.debug:
        #     # Draw lines on image
        #     lines_image = image_msg.copy()

        #     def plot_line(rho, theta, color=(0.0, 1.0, 0.0)):
        #         # RGB to BGR
        #         color = (color[2], color[1], color[0])
        #         color = tuple(int(c * 255) for c in color)
        #         a = math.cos(theta)
        #         b = math.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        #         cv.line(lines_image, pt1, pt2, color, 2, cv.LINE_AA)

        #     left_lines, right_lines = get_all_lanes(self.hough_line_params, white_line_mask)
        #     raw_lines = get_raw_lines(self.hough_line_params, white_line_mask)
        #     # self.get_logger().info(f"Left lines: {left_lines}")
        #     # self.get_logger().info(f"Right lines: {right_lines}")
        #     for rho, theta in left_lines:
        #         plot_line(rho, theta, color=self.left_lane_color)
        #     for rho, theta in right_lines:
        #         plot_line(rho, theta, color=self.right_lane_color)

        #     left_line, right_line = get_best_lanes(self.hough_line_params, white_line_mask)
        #     if left_line and right_line:
        #         mid_line = find_mid_line(left_line, right_line)
        #         plot_line(mid_line[0], mid_line[1], color=self.midline_color)

        #     # plot_line(200, 0, color=(0.5, 0.5, 0.5))  # 0 is vertical
        #     # plot_line(200, np.pi / 2, color=(1.0, 0.0, 0.0))  # np.pi / 2 is horizontal

        #     debug_msg = self.bridge.cv2_to_imgmsg(lines_image, "bgr8")
        #     # filtered_image = cv2.bitwise_and(image, image, mask=white_line_mask)
        #     # debug_msg = self.bridge.cv2_to_imgmsg(filtered_image, "bgr8")
        #     self.debug_image_pub.publish(debug_msg)


    def pub_visualization_and_trajectory(self, left_line_candidates, right_line_candidates):
        """
        Process detected lane lines, transform to world coordinates, and publish
        the trajectory and visualization markers.

        Args:
            left_line_candidates (numpy.ndarray): Detected left lane line segments
            right_line_candidates (numpy.ndarray): Detected right lane line segments

        Returns:
            None
        """
        left_line_pxls = self.choose_line(left_line_candidates)
        right_line_pxls = self.choose_line(right_line_candidates)

        trajectory_pxls = self.get_trajectory_pxls(left_line_pxls, right_line_pxls)

        left_line_relxy = self.homography_transformer.transformLine(left_line_pxls[:, :4])
        right_line_relxy = self.homography_transformer.transformLine(right_line_pxls[:, :4])
        trajectory_relxy = self.homography_transformer.transformLine(trajectory_pxls[:, :4])

        # VisualizationTools.plot_line(left_line_relxy, color=self.left_lane_color, frame="base_link")
        # VisualizationTools.plot_line(right_line_relxy, color=self.right_lane_color, frame="base_link")
        # VisualizationTools.plot_line(trajectory_relxy, color=self.midline_color, frame="base_link")

        VisualizationTools.plot_line(
            left_line_relxy[:, 0],  # x coordinates
            left_line_relxy[:, 1],  # y coordinates
            self.left_lane_pub,     # publisher
            color=self.left_lane_color,
            frame="base_link"
        )

        VisualizationTools.plot_line(
            right_line_relxy[:, 0],  # x coordinates
            right_line_relxy[:, 1],  # y coordinates
            self.right_lane_pub,    # publisher
            color=self.right_lane_color,
            frame="base_link"
        )

        VisualizationTools.plot_line(
            trajectory_relxy[:, 0],  # x coordinates
            trajectory_relxy[:, 1],  # y coordinates
            self.midline_visualization_pub,  # publisher
            color=self.midline_color,
            frame="base_link"
        )

        self.publish_trajectory(trajectory_relxy)

    # def choose_line(self, line_candidates):
    #     """
    #     Select a representative line from multiple candidates by filtering and merging.

    #     Args:
    #         line_candidates (numpy.ndarray): Array of line candidates

    #     Returns:
    #         numpy.ndarray: Best line parameters
    #     """
    #     if len(line_candidates) == 0:
    #         return None

    #     # Get image dimensions
    #     image_height = self.image_height
    #     image_width = self.image_width

    #     # Filter lines to be below the middle of the image
    #     row_threshold = image_height / 2
    #     filtered_lines = filter_line_candidates_by_row_extent(line_candidates, row_threshold)

    #     if len(filtered_lines) == 0:
    #         return line_candidates[0:1]  # Return first line if no lines pass the filter

    #     # Cluster and merge lines with similar x-intercepts
    #     merged_lines = cluster_and_merge_lines(filtered_lines)

    #     # Select the best line based on distance from center
    #     best_line = select_best_line_candidate(merged_lines, image_width)

    #     if best_line is None and len(line_candidates) > 0:
    #         return line_candidates[0:1]  # Fallback

    #     return best_line

    def choose_line(self, line_candidates):
        """
        Select a representative line from multiple candidates by filtering and merging.

        Args:
            line_candidates (numpy.ndarray): Array of line candidates

        Returns:
            numpy.ndarray: Best line parameters
        """
        # return select_best_line_candidate(line_candidates, self.image_width, self.image_height)
        return select_best_line_candidate_by_midpoint_radius(line_candidates, self.image_width, self.image_height)


    def get_trajectory_pxls(self, left_line, right_line):
        """
        Calculate the midline trajectory between the left and right lane lines.

        Args:
            left_line (numpy.ndarray): Left lane line parameters
            right_line (numpy.ndarray): Right lane line parameters

        Returns:
            numpy.ndarray: Midline trajectory parameters
        """
        left_line_trimmed = left_line[:, :4]
        right_line_trimmed = right_line[:, :4]
        lanes_midline = 1/2 * (left_line_trimmed + right_line_trimmed)
        return lanes_midline
    # def publish_trajectory(self, trajectory_points):
    #     """
    #     Publish the trajectory message for the lane midline.

    #     Args:
    #         trajectory_points (numpy.ndarray): Trajectory points in world coordinates

    #     Returns:
    #         None
    #     """
    #     trajectory_msg = Trajectory()

    #     # Create points from trajectory line
    #     start_point = Point()
    #     end_point = Point()

    #     start_point.x = float(trajectory_points[0][0])  # x1
    #     start_point.y = float(trajectory_points[0][1])  # y1

    #     end_point.x = float(trajectory_points[1][0])    # x2
    #     end_point.y = float(trajectory_points[1][1])    # y2

    #     trajectory_msg.points = [start_point, end_point]
    #     self.trajectory_pub.publish(trajectory_msg)

    def publish_trajectory(self, trajectory_points):
        """
        Publish the trajectory message for the lane midline.

        Args:
            trajectory_points (numpy.ndarray): Trajectory points in world coordinates

        Returns:
            None
        """
        try:
            trajectory_msg = Trajectory()

            # Create points from trajectory line
            start_point = Point()
            end_point = Point()

            start_point.x = float(trajectory_points[0][0])  # x1
            start_point.y = float(trajectory_points[0][1])  # y1

            end_point.x = float(trajectory_points[1][0])    # x2
            end_point.y = float(trajectory_points[1][1])    # y2

            trajectory_msg.points = [start_point, end_point]
            self.trajectory_pub.publish(trajectory_msg)
        except Exception as e:
            self.get_logger().error(f"Error in publishing trajectory: {e}")

def main(args=None):
    rclpy.init(args=args)
    lane_detector = LaneDetector()
    rclpy.spin(lane_detector)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
