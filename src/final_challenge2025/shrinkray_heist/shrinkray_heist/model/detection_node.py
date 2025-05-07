import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
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

        if not self.is_sim:
            self.detector = Detector()
            self.detector.set_threshold(0.25)
            self.debug_image_pub = self.create_publisher(Image, "/debug_image", 10)
            self.subscriber = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.callback, 1)
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

        self.get_logger().info("Detector Initialized")

    def callback(self, img_msg):
        if self.is_sim:
            # Just always put a banana
            self.out_tf_mut.header.stamp = self.get_clock().now().to_msg()
            self.out_tf_mut.transform.translation.x = 1.0
            self.out_tf_mut.transform.translation.y = -0.5
            self.broadcaster.sendTransform(self.out_tf_mut)
            return

        # Process image with CV Bridge
        image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

        results = self.detector.predict(image)

        predictions = results["predictions"]
        original_image = results["original_image"]

        banana_bounding_boxes = [point for point, label in predictions if label in ("banana", "frisbee")]

        if banana_bounding_boxes:
            xmin, ymin, xmax, ymax = banana_bounding_boxes[0]
            x, y = ((xmin + xmax) / 2, ymax)
            x, y = homography_utils.transform_uv_to_xy(x, y)

            self.out_tf_mut.header.stamp = self.get_clock().now().to_msg()

            self.out_tf_mut.transform.translation.x = x
            self.out_tf_mut.transform.translation.y = y
            self.broadcaster.sendTransform(self.out_tf_mut)
        else:
            pass

        if self.debug:
            boxed_img = self.detector.draw_box(original_image, predictions, draw_all=True)
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
