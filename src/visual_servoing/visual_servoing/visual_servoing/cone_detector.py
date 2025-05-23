#!/usr/bin/env python

import cv2
import numpy as np
import rclpy

# import your color segmentation algorithm; call this function in ros_image_callback!
from computer_vision.color_segmentation import cd_color_segmentation, image_print
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point  # geometry_msgs not in CMake file
from rclpy.node import Node
from sensor_msgs.msg import Image
from vs_msgs.msg import ConeLocationPixel


class ConeDetector(Node):
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """

    def __init__(self):
        super().__init__("cone_detector")
        # Toggle line follower vs cone parker
        self.line_following = self.declare_parameter("line_following", False).get_parameter_value().bool_value

        # Subscribe to ZED camera RGB frames
        self.cone_pub = self.create_publisher(ConeLocationPixel, "/relative_cone_px", 10)
        self.debug_pub = self.create_publisher(Image, "/cone_debug_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge()  # Converts between ROS images and OpenCV Images

        self.get_logger().info("Cone Detector Initialized")

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
        # Extract image from ROS message
        # img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        bounding_box, hsv = cd_color_segmentation(image, None, line_following=self.line_following)
        if bounding_box is not None:
            print("Bounding box: ", bounding_box)
            u = (bounding_box[0][0] + bounding_box[1][0]) // 2
            v = bounding_box[1][1]
            cone_msg = ConeLocationPixel()
            cone_msg.u = float(u)
            cone_msg.v = float(v)
            self.cone_pub.publish(cone_msg)
            cv2.rectangle(hsv, *bounding_box, color=[0, 255, 0], thickness=2)
            # image_print(image)
        debug_msg = self.bridge.cv2_to_imgmsg(hsv, "bgr8")
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    cone_detector = ConeDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
