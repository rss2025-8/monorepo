#!/usr/bin/env python

import cv2 as cv
import numpy as np
import rclpy

from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from vs_msgs.msg import Point, Trajectory
from race_to_the_moon.computer_vision.color_segmentation import bgr_color_segmentation
from race_to_the_moon.visualization_tools import VisualizationTools
from race_to_the_moon.computer_vision.homography_utils import HomographyTransformer

class LaneDetector(Node):
    """
    A class for lane detection algorithms
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /trajectory (Trajectory).
    """

    def __init__(self):
        super().__init__("lane_detector")

        self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')

        # Additional parameter declarations
        self.declare_parameter('left_lane_color', [0.0, 1.0, 0.0])
        self.declare_parameter('right_lane_color', [1.0, 0.0, 0.0])
        self.declare_parameter('midline_color', [0.0, 0.0, 1.0])
        self.declare_parameter('use_homography', 'rss2025_8_old')
        self.declare_parameter('meters_per_inch', 0.0254)
        self.declare_parameter('visualization', True)

        # Color segmentation parameters
        self.declare_parameter('min_value', 120.)
        self.declare_parameter('max_saturation', 25.)

        # Hough line parameters declaration
        self.declare_parameter('rho_resolution', 2)
        self.declare_parameter('theta_resolution', np.pi/180)
        self.declare_parameter('threshold', 100)
        self.declare_parameter('minLineLength', 20)
        self.declare_parameter('maxLineGap', 50)
        self.declare_parameter('canny_low', 50)
        self.declare_parameter('canny_high', 150)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 360)
        self.declare_parameter("debug", True)

        # Homography parameters
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
        self.PTS_IMAGE_PLANE = self.get_parameter(f'HOMOGRAPHIES.{homography_name}.PTS_IMAGE_PLANE').value
        self.PTS_GROUND_PLANE = self.get_parameter(f'HOMOGRAPHIES.{homography_name}.PTS_GROUND_PLANE').value
        self.METERS_PER_INCH = self.get_parameter('meters_per_inch').value
        self.image_topic = self.get_parameter('image_topic').value

        # Get parameters
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.left_lane_color = self.get_parameter('left_lane_color').value
        self.right_lane_color = self.get_parameter('right_lane_color').value
        self.midline_color = self.get_parameter('midline_color').value
        self.visualization = self.get_parameter('visualization').value
        self.debug = self.get_parameter("debug").value

        self.color_segmentation_params = {
            "min_value": self.get_parameter("min_value").value,
            "max_saturation": self.get_parameter("max_saturation").value,
            }

        # Create subscriptions and publishers
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 5)
        self.bridge = CvBridge()

        self.left_lane_pub = self.create_publisher(Marker, "/race/left_lane", 10)
        self.right_lane_pub = self.create_publisher(Marker, "/race/right_lane", 10)
        self.midline_visualization_pub = self.create_publisher(Marker, "/race/mid_lane", 10)
        self.trajectory_pub = self.create_publisher(Trajectory, "/race/trajectory", 10)
        self.debug_image_pub = self.create_publisher(Image, "/race/debug_img", 10)

        # Configure Hough line detection parameters
        self.hough_line_params = {
            "rho_resolution": self.get_parameter('rho_resolution').value,
            "theta_resolution": self.get_parameter('theta_resolution').value,
            "threshold": self.get_parameter('threshold').value,
            "minLineLength": self.get_parameter('minLineLength').value,
            "maxLineGap": self.get_parameter('maxLineGap').value,
            "canny_low": self.get_parameter('canny_low').value,
            "canny_high": self.get_parameter('canny_high').value,
        }

        self.line_filtering_params = {
            "max_left_angle": -np.pi / 12,
            "min_left_angle": -11*np.pi / 12,
            "max_right_angle": 11*np.pi / 12,
            "min_right_angle": np.pi/12
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

    def get_canny(self, image):
        """Apply Canny edge detection to the image"""
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        canny = cv.Canny(blur, self.hough_line_params["canny_low"], self.hough_line_params["canny_high"])
        return canny

    def crop_image(self, image):
        """Isolate the region of interest (ROI) using a rectangular mask"""
        height = image.shape[0]
        width = image.shape[1]
        # Create a rectangular region of interest
        polygons = np.array([
            [(0, height), (width, height), (width, height//2), (0, height//2)]
        ])
        mask = np.zeros_like(image)
        cv.fillPoly(mask, polygons, (255, 255, 255))
        masked_image = cv.bitwise_and(image, mask)
        return masked_image

    def make_coordinates(self, image, line_parameters):
        """Convert line parameters to x,y coordinates"""
        try:
            slope, intercept = line_parameters
            y1 = image.shape[0]
            y2 = int(y1*(3/5))

            # Check if slope is valid
            if abs(slope) < 0.001:  # Avoid near-horizontal lines
                return None

            x1 = int((y1 - intercept)/slope)
            x2 = int((y2 - intercept)/slope)

            # Ensure coordinates are within image bounds
            height, width = image.shape[:2]
            x1 = max(0, min(x1, width-1))
            x2 = max(0, min(x2, width-1))

            return np.array([x1, y1, x2, y2])
        except:
            return None

    def average_slope_intercept(self, image, lines):
        """Calculate average slope and intercept for left and right lanes"""
        left_fit = []
        right_fit = []
        if lines is None:
            return None

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # Check if line is not vertical (avoid division by zero)
            if abs(x2 - x1) > 0.1:  # Avoid very small differences
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope = parameters[0]
                intercept = parameters[1]
                # Filter out unrealistic slopes
                if abs(slope) > 0.1 and abs(slope) < 10:
                    # Separate left and right lanes based on slope
                    if self.line_filtering_params["min_left_angle"] < slope < self.line_filtering_params["max_left_angle"] and slope:
                        left_fit.append((slope, intercept))
                    elif self.line_filtering_params["min_right_angle"] < slope < self.line_filtering_params["max_right_angle"] and slope:
                        right_fit.append((slope, intercept))
                    else:
                        pass

        lines_to_draw = []
        if len(left_fit) > 0:
            left_fit_average = np.average(left_fit, axis=0)
            left_line = self.make_coordinates(image, left_fit_average)
            if left_line is not None and not np.isnan(left_line).any():
                lines_to_draw.append(left_line)

        if len(right_fit) > 0:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = self.make_coordinates(image, right_fit_average)
            if right_line is not None and not np.isnan(right_line).any():
                lines_to_draw.append(right_line)

        if len(lines_to_draw) > 0:
            return np.array(lines_to_draw)
        return None

    def display_lines(self, image, lines):
        """Draw the detected lines on the image"""
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            avg_line = np.mean(lines, axis=0).astype(int)
            x1, y1, x2, y2 = avg_line
            cv.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
        return line_image

    # def temp_display_midline(self, image, lines):
    #     """Draw the midline based on the average of detected lines"""
    #     line_image = np.zeros_like(image)
    #     if lines is not None and len(lines) > 0:
    #         avg_line = np.mean(lines, axis=0).astype(int)
    #         x1, y1, x2, y2 = avg_line
    #         cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    #     return line_image

    def image_callback(self, image_msg):
        bgr_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        if bgr_image is None:
            return

        cropped_image = self.crop_image(bgr_image)

        cropped_and_color_segmented_image = bgr_color_segmentation(cropped_image, self.color_segmentation_params["min_value"], self.color_segmentation_params["max_saturation"])

        # Step 1: Apply Canny edge detection
        # canny_image = self.canny_detection(bgr_image)
        canny_image = self.get_canny(cropped_and_color_segmented_image)

        # Step 2: Define region of interest
        # cropped_image = self.region_of_interest(canny_image)

        # Step 3: Apply Hough Transform
        lines = cv.HoughLinesP(
            canny_image,
            self.hough_line_params['rho_resolution'],
            self.hough_line_params['theta_resolution'],
            self.hough_line_params['threshold'],
            minLineLength=self.hough_line_params['minLineLength'],
            maxLineGap=self.hough_line_params['maxLineGap']
        )

        # Step 4: Average and extrapolate lanes
        averaged_lines = self.average_slope_intercept(bgr_image, lines)

        if averaged_lines is not None:
            # Process detected lanes
            left_line = None
            right_line = None

            for line in averaged_lines:
                x1, y1, x2, y2 = line.reshape(4)
                # Determine if line is left or right based on position and slope
                if abs(x2 - x1) > 0.1:
                    slope = (y2 - y1) / (x2 - x1)
                    if slope < 0:  # Left lane has negative slope
                        left_line = line.reshape(1, 4)
                    else:  # Right lane has positive slope
                        right_line = line.reshape(1, 4)

            # Handle lane line detection scenarios
            if left_line is not None and right_line is not None:
                self.pub_visualization_and_trajectory(left_line, right_line)
            elif left_line is not None:
                # No right line detected - create virtual right line
                right_boundary_line = np.array([[self.image_width, 0, self.image_width, self.image_height]])
                self.pub_visualization_and_trajectory(left_line, right_boundary_line)
            elif right_line is not None:
                # No left line detected - create virtual left line
                left_boundary_line = np.array([[0, 0, 0, self.image_height]])
                self.pub_visualization_and_trajectory(left_boundary_line, right_line)
            else:
                self.get_logger().warning("No lane lines detected")
        else:
            self.get_logger().warning("No lanes detected in Hough Transform")

        if self.debug:
            image_param = cropped_and_color_segmented_image #cv.cvtColor(canny_image, cv.COLOR_GRAY2BGR)
            # Draw lines on image
            line_image = self.display_lines(image_param, averaged_lines)
            # line_image = self.temp_display_midline(line_image, averaged_lines)

            # Combine with original image
            if averaged_lines is not None:
                combo_image = cv.addWeighted(image_param, 0.8, line_image, 1, 1)
            else:
                combo_image = image_param

            debug_msg = self.bridge.cv2_to_imgmsg(combo_image, "bgr8")
            self.debug_image_pub.publish(debug_msg)

    def pub_visualization_and_trajectory(self, left_line_candidates, right_line_candidates):
        """
        Process detected lane lines, transform to world coordinates, and publish
        the trajectory and visualization markers.
        """
        # trajectory_pxls = self.get_trajectory_pxls(left_line_candidates, right_line_candidates)

        if self.visualization:
            left_line_relxy = self.homography_transformer.transformLine(left_line_candidates)
            right_line_relxy = self.homography_transformer.transformLine(right_line_candidates)
        trajectory_relxy = 1/2 * (left_line_relxy + right_line_relxy)
        # trajectory_relxy = self.homography_transformer.transformLine(trajectory_pxls)

        if self.visualization:
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

    def get_trajectory_pxls(self, left_line, right_line):
        """
        Calculate the midline trajectory between the left and right lane lines.
        """
        midline_y1 = self.image_height
        midline_y2 = 0

        midline = 1/2 * (left_line + right_line)  # average the endpoints
        extended_midline = self.extend_line(midline, midline_y1, midline_y2)

        return extended_midline

    def extend_line(self, line: np.ndarray, desired_y1: int, desired_y2: int) -> np.ndarray:
        """Extend a line to reach specified y coordinates"""
        assert line.shape == (1, 4)
        x1, y1, x2, y2 = line[0, 0], line[0, 1], line[0, 2], line[0, 3]

        if abs(x2 - x1) < 0.001:  # Avoid division by zero
            return line

        m_line = (y2 - y1) / (x2 - x1)
        desired_x1 = ((desired_y1 - y1) / m_line) + x1
        desired_x2 = ((desired_y2 - y1) / m_line) + x1
        return np.array([[desired_x1, desired_y1, desired_x2, desired_y2]])

    def publish_trajectory(self, trajectory_points):
        """Publish the trajectory message for the lane midline."""
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
