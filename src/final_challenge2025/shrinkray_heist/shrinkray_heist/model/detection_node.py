import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker

from .. import visualize
from . import homography_utils
from .detector import Detector


class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector")
        self.is_sim = self.declare_parameter("is_sim", False).value
        self.debug = True
        self.update_period: int = self.declare_parameter("update_period", 3).value
        self.update_period_counter: int = 0
        self.threshold = 0.1
        self.lower_threshold = 0.01

        if not self.is_sim:
            self.detector = Detector()
            self.detector.set_threshold(self.threshold)
            self.debug_image_pub = self.create_publisher(Image, "/debug_image", 10)
            self.subscriber = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.callback, 1)
            # self.homography_sub = self.create_subscription(Point, "/zed/zed_node/left/image_rect_color_mouse_left", self.homography_callback, 1)
            self.bridge = CvBridge()

            self.out_tf_mut = TransformStamped()
            self.out_tf_mut.header.frame_id = "base_link"  # Homography is already in base_link frame
            self.out_tf_mut.child_frame_id = "banana"

            self.broadcaster = TransformBroadcaster(self)
        else:
            self.get_logger().info("Running in simulation mode...")
            self.subscriber = self.create_subscription(Odometry, "/pf/pose/odom", self.callback, 1)
            self.out_tf_mut = TransformStamped()
            self.out_tf_mut.header.frame_id = "base_link"  # Homography is already in base_link frame
            self.out_tf_mut.child_frame_id = "banana"

            self.broadcaster = TransformBroadcaster(self)

        self.banana_state_sub = self.create_subscription(Bool, "/i_hate_ros", self.seen_banana_update, 1)
        self.seen_banana = False

        self.get_logger().info("Detector Initialized")

    # def homography_callback(self, msg: Point) -> None:
    #     # Convert to xy
    #     x, y = homography_utils.transform_uv_to_xy(msg.x, msg.y)
    #     self.get_logger().info(f"({msg.x}, {msg.y}) -> Homography: {x:.2f} m, {y:.2f} m")

    def seen_banana_update(self, msg: Bool) -> None:
        self.seen_banana = True

    def callback(self, img_msg):
        if self.is_sim:
            # Put a banana sometimes
            if np.random.rand() < 0.1:
                self.out_tf_mut.header.stamp = self.get_clock().now().to_msg()
                self.out_tf_mut.transform.translation.x = np.random.uniform(0.7, 1.3)
                self.out_tf_mut.transform.translation.y = np.random.uniform(-0.5, 0.5)
                self.broadcaster.sendTransform(self.out_tf_mut)
            return

        # Update every `update_period` frames
        self.update_period_counter += 1
        if self.update_period_counter >= self.update_period:
            self.update_period_counter = 0
        else:
            return

        # Process image with CV Bridge
        image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

        self.detector.set_threshold(self.threshold if not self.seen_banana else self.lower_threshold)
        results = self.detector.predict(image)

        predictions = results["predictions"]
        # Sort by confidence
        predictions.sort(key=lambda x: x[2], reverse=True)
        original_image = results["original_image"]

        banana_bounding_boxes = [
            point for point, label, confidence in predictions if label == "banana" and confidence >= self.threshold
        ]

        if banana_bounding_boxes:
            xmin, ymin, xmax, ymax = banana_bounding_boxes[0]  # Highest confidence prediction
            x, y = ((xmin + xmax) / 2, ymax)
            x, y = homography_utils.transform_uv_to_xy(x, y)

            self.out_tf_mut.header.stamp = self.get_clock().now().to_msg()

            self.out_tf_mut.transform.translation.x = x
            self.out_tf_mut.transform.translation.y = y
            self.broadcaster.sendTransform(self.out_tf_mut)
        else:
            pass

        # save image
        if self.seen_banana and banana_bounding_boxes:
            self.seen_banana = False
            # Draw only the first banana bounding box
            banana_predictions = [(banana_bounding_boxes[0], "banana")]
            banana_img = self.detector.draw_box(original_image, banana_predictions, draw_all=True)
            save_path = f"{os.path.dirname(__file__)}/banana_{self.get_clock().now().nanoseconds}.png"
            banana_img.save(save_path)
            self.get_logger().info(f"Saved banana image to {save_path}")

        if self.debug:
            boxed_img = self.detector.draw_box(original_image, [(point, label) for point, label, _ in predictions], draw_all=True)
            cv2_img = cv2.cvtColor(np.array(boxed_img), cv2.COLOR_RGB2BGR)
            debug_img = self.bridge.cv2_to_imgmsg(cv2_img, "bgr8")

            self.debug_image_pub.publish(debug_img)

        # self.get_logger().info(f"bananas: {banana_bounding_boxes}")
        # self.


def main(args=None):
    rclpy.init(args=args)
    detector = DetectorNode()
    rclpy.spin(detector)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
