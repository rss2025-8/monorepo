#!/usr/bin/env python
"""Visualizes output of modules."""

import cv2
import numpy as np
import race_to_the_moon.visualize as visualize
import rclpy
import tf2_ros
import tf_transformations
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, Pose, Quaternion, TransformStamped, Vector3
from race_to_the_moon.homography import transform_uv_to_xy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from vs_msgs.msg import LookaheadLocation


def point_to_pose(x, y, theta) -> Pose:
    """Convert (x, y, theta) to a Pose message."""
    quaternion = tf_transformations.quaternion_about_axis(theta, (0, 0, 1))
    return Pose(
        position=Point(x=x, y=y, z=0.0),
        orientation=Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]),
    )


def pose_to_tf(pose: Pose, parent: str, child: str, time: rclpy.time.Time) -> TransformStamped:
    """Converts a rotation matrix to a TransformStamped message.

    Trailing parameters follow the same format as TF's lookup_transform().
    """
    header = Header(stamp=time.to_msg(), frame_id=parent)
    msg = TransformStamped(header=header, child_frame_id=child)
    # Setup rotation and translation portions of transform
    msg.transform.rotation = Quaternion(
        x=pose.orientation.x, y=pose.orientation.y, z=pose.orientation.z, w=pose.orientation.w
    )
    msg.transform.translation = Vector3(x=pose.position.x, y=pose.position.y, z=pose.position.z)
    return msg


class VisualizerNode(Node):

    def __init__(self):
        super().__init__("visualizer")

        self.image_topic: str = self.declare_parameter("image_topic", "/zed/zed_node/rgb/image_rect_color").value
        # self.image_topic: str = self.declare_parameter("image_topic", "/race/debug_img").value
        self.debug: bool = self.declare_parameter("debug", True).value

        self.image_pub = self.create_publisher(Marker, "/race/flat_image", 1)
        self.bridge = CvBridge()

        self.static_br = tf2_ros.StaticTransformBroadcaster(self)  # Broadcast static TF transform
        map_to_base_link = pose_to_tf(point_to_pose(0.0, 0.0, 0.0), "map", "base_link", self.get_clock().now())
        self.static_br.sendTransform([map_to_base_link])

        if self.debug:
            self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 1)
            self.get_logger().info("DEBUG mode enabled")
        self.get_logger().info("Visualizer initialized")

    def image_callback(self, image_msg):
        if self.debug:
            image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            visualize.plot_image(image, self.image_pub, scale=0.08, sample_shape=(360 // 2, 640 // 20))


def main(args=None):
    rclpy.init(args=args)
    visualizer_node = VisualizerNode()
    rclpy.spin(visualizer_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
