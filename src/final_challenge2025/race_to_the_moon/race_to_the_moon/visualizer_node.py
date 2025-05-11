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
from sensor_msgs.msg import CompressedImage, Image
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


img_counter = 0


class VisualizerNode(Node):

    def __init__(self):
        super().__init__("visualizer_node")

        self.image_topic: str = self.declare_parameter("image_topic", "/zed/zed_node/rgb/image_rect_color").value
        # self.image_topic: str = self.declare_parameter("image_topic", "/race/debug_img").value
        self.compressed_image_topic: str = self.declare_parameter(
            "compressed_image_topic", "/zed/zed_node/rgb/image_rect_color/compressed"
        ).value
        self.debug: bool = self.declare_parameter("debug", True).value
        self.fast_mode: bool = self.declare_parameter("fast_mode", False).value

        self.image_pub = self.create_publisher(Marker, "/race/flat_image", 1)
        self.bridge = CvBridge()

        self.static_br = tf2_ros.StaticTransformBroadcaster(self)  # Broadcast static TF transform
        map_to_base_link = pose_to_tf(point_to_pose(0.0, 0.0, 0.0), "map", "base_link", self.get_clock().now())
        self.static_br.sendTransform([map_to_base_link])

        # Track runtime
        self.timing = [0, 0.0]

        if self.debug:
            self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 1)
            self.compressed_image_sub = self.create_subscription(
                CompressedImage, self.compressed_image_topic, self.compressed_image_callback, 1
            )
            self.decompressed_image_pub = self.create_publisher(Image, "/zed/zed_node/rgb/image_rect_color", 1)
            self.get_logger().info("DEBUG mode enabled")
        if self.fast_mode:
            self.get_logger().info("FAST mode enabled")
        self.get_logger().info("Visualizer initialized")

    def image_callback(self, image_msg):
        # Save to image with increasing timestamp
        # image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        # global img_counter
        # cv2.imwrite(f"images/{img_counter}.png", image)
        # img_counter += 1
        # return

        if self.debug:
            call_time = self.get_clock().now()

            image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            if self.fast_mode:
                visualize.plot_image(image, self.image_pub, scale=0.16, sample_shape=(360 // 10, 640 // 20))
            else:
                visualize.plot_image(image, self.image_pub, scale=0.08, sample_shape=(360 // 5, 640 // 10))
                # visualize.plot_image(image, self.image_pub, scale=0.08, sample_shape=(360 // 2, 640 // 20))

            latency = (self.get_clock().now() - call_time).nanoseconds / 1e9
            self.timing[0] += 1
            self.timing[1] += latency
            if self.timing[0] == 50:
                avg_latency = self.timing[1] / 50
                if avg_latency > 0.02:
                    self.get_logger().warning(f"high visualizer latency, optimize: {avg_latency:.4f}s")
                else:
                    self.get_logger().info(f"visual: {avg_latency:.4f}s")
                self.timing = [0, 0.0]

    def compressed_image_callback(self, compressed_image_msg):
        # Uncompress image and republish
        cv_image = self.bridge.compressed_imgmsg_to_cv2(compressed_image_msg, desired_encoding="bgr8")
        raw_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        raw_msg.header = compressed_image_msg.header
        self.decompressed_image_pub.publish(raw_msg)


def main(args=None):
    rclpy.init(args=args)
    visualizer_node = VisualizerNode()
    rclpy.spin(visualizer_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
