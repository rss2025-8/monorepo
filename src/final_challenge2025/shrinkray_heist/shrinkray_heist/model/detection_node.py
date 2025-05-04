from shrinkray_heist import homography_utils
import numpy as np
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseStamped
from .detector import Detector

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector")
        self.debug = True

        self.detector = Detector()
        self.detector.set_threshold(0.25)

        self.point_pub = self.create_publisher(PoseStamped, "/detected_point", 10)
        self.debug_image_pub = self.create_publisher(Image, "/debug_image", 10)
        self.subscriber = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.callback, 1)
        self.bridge = CvBridge()

        self.out_tf_mut = TransformStamped()
        self.out_tf_mut.header.frame_id = "zed_camera_center"
        self.out_tf_mut.child_frame_id = "banana"

        self.broadcaster = TransformBroadcaster(self)

        self.get_logger().info("Detector Initialized")

    def callback(self, img_msg):
        # Process image with CV Bridge
        image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

        results = self.detector.predict(image)

        predictions = results["predictions"]
        original_image = results["original_image"]

        banana_bounding_boxes = [point for point, label in predictions if label in ("banana", "frisbee")]

        if banana_bounding_boxes:
            xmin, ymin, xmax, ymax = banana_bounding_boxes[0]
            x, y = ((xmin + xmax) / 2, (ymin + ymax) / 2)
            x, y = homography_utils.transform_uv_to_xy(x, y)

            self.out_tf_mut.header.stamp = self.get_clock().now().to_msg()

            self.out_tf_mut.transform.translation.x = x
            self.out_tf_mut.transform.translation.y = y

            self.broadcaster.sendTransform(self.out_tf_mut)
        else:
            pass


        if self.debug:
          boxed_img = self.detector.draw_box(original_image, predictions, draw_all=True)
          debug_img = self.bridge.cv2_to_imgmsg(np.array(boxed_img), "bgr8")

          self.debug_image_pub.publish(debug_img)

        self.get_logger().info(f"bananas: {banana_bounding_boxes}")
        # self.

def main(args=None):
    rclpy.init(args=args)
    detector = DetectorNode()
    rclpy.spin(detector)
    rclpy.shutdown()

if __name__=="__main__":
    main()
