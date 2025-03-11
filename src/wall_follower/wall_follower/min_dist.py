import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np

class LaserScanListener(Node):
    def __init__(self):
        super().__init__('laser_scan_listener')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        self.subscription  # Prevent unused variable warning
        self.global_min_distance = float('inf')
        self.avg_min_distance = float('inf')
        self.side = "left"

    def scan_callback(self, msg):
        """ Extracts the minimum distance from the right side of the LaserScan message."""
        num_readings = len(msg.ranges)
        if self.side == "right":
            right_side_ranges = msg.ranges[:num_readings // 2]  # Assuming right side is the first half
            valid_ranges = [r for r in right_side_ranges if r > msg.range_min and r < msg.range_max]
            min_distance = min(valid_ranges) if valid_ranges else float('inf')
            self.global_min_distance = min(self.global_min_distance, min_distance)
            self.get_logger().info(f"Minimum distance to right wall: {min_distance:.3f} meters")
            self.get_logger().info(f"Global minimum distance to right wall: {self.global_min_distance:.3f} meters")
            self.avg_min_distance = np.mean([self.global_min_distance, min_distance])
            self.get_logger().info(f"Average minimum distance to right wall: {self.avg_min_distance:.3f} meters")
        else:
            left_side_ranges = msg.ranges[num_readings // 2:]
            valid_ranges = [r for r in left_side_ranges if r > msg.range_min and r < msg.range_max]
            min_distance = min(valid_ranges) if valid_ranges else float('inf')
            self.global_min_distance = min(self.global_min_distance, min_distance)
            self.get_logger().info(f"Minimum distance to left wall: {min_distance:.3f} meters")
            self.get_logger().info(f"Global minimum distance to left wall: {self.global_min_distance:.3f} meters")
            self.avg_min_distance = np.mean([self.global_min_distance, min_distance])
            self.get_logger().info(f"Average minimum distance to left wall: {self.avg_min_distance:.3f} meters")

def main(args=None):
    rclpy.init(args=args)
    node = LaserScanListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
