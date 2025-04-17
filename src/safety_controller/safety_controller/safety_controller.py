import math

import numpy as np
import rclpy
import rclpy.time
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Header
from visualization_msgs.msg import Marker


def scan_index(scan: LaserScan, angle: float) -> int:
    return int((angle - scan.angle_min) // scan.angle_increment)


def time_to_seconds(time: Time) -> float:
    return time.nanoseconds / rclpy.time.CONVERSION_CONSTANT


def delta_time(start: Time, end: Time) -> float:
    return time_to_seconds(end) - time_to_seconds(start)


# todo deal with signs
# 0 = v t + 1/2 a t^2 - dist
# t = - v +- sqrt(v^2 - 4 dist a) / -2 dist


class SafetyController(Node):

    def __init__(self):
        super().__init__("safety_controller")

        scan_topic: str = self.declare_parameter("scan_topic", "scan").get_parameter_value().string_value
        ackermann_cmd_topic: str = self.declare_parameter("ackermann_cmd_topic", "vesc/low_level/ackermann_cmd").value
        safety_topic: str = self.declare_parameter("safety_topic", "vesc/low_level/input/safety").value
        self.watchdog_localize_topic: str = self.declare_parameter("watchdog_localize_topic", "").value

        self.stopping_time: float = self.declare_parameter("stopping_time", 0.7).value
        self.watchdog_period: float = self.declare_parameter("watchdog_period", 0.5).value
        self.min_front_distance: float = self.declare_parameter("min_front_distance", 0.5).value
        self.min_side_distance: float = self.declare_parameter("min_side_distance", 0.25).value

        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.ackermann_sub = self.create_subscription(
            AckermannDriveStamped, ackermann_cmd_topic, self.ackermann_cmd_callback, 10
        )
        self.safety_pub = self.create_publisher(AckermannDriveStamped, safety_topic, 10)

        self.timer = self.create_timer(0.05, self.timer_callback)  # Make sure safety still runs without other callbacks

        self.ackermann_drive_msg_mut: AckermannDriveStamped = AckermannDriveStamped()

        if self.watchdog_localize_topic:
            self.localize_timestamp = self.get_clock().now()
            self.localize_sub = self.create_subscription(
                Odometry, self.watchdog_localize_topic, self.localize_callback, 10
            )

        self.scan_timestamp = self.get_clock().now()
        self.ackermann_timestamp = self.get_clock().now()
        self.last_debug_timestamp = self.get_clock().now()
        self.front_distance = float("inf")
        self.left_distance = float("inf")
        self.right_distance = float("inf")
        self.speed = 0.0
        self.steering_angle = 0.0

        self.get_logger().info("Safety controller initialized")

    def localize_callback(self, odom: Odometry) -> None:
        # Mark that we received a pose estimate
        self.localize_timestamp = Time.from_msg(odom.header.stamp)

    def scan_callback(self, scan: LaserScan) -> None:
        # Mark that we received a scan
        self.scan_timestamp = Time.from_msg(scan.header.stamp)
        min_index = scan_index(scan, -math.pi / 12)
        max_index = scan_index(scan, math.pi / 12)
        self.front_distance = min(scan.ranges[min_index : max_index + 1])
        self.right_distance = min(scan.ranges[:min_index])
        self.left_distance = min(scan.ranges[max_index + 1 :])
        self.timer_callback()  # Check safety conditions

    def ackermann_cmd_callback(self, cmd: AckermannDriveStamped) -> None:
        self.ackermann_timestamp = Time.from_msg(cmd.header.stamp)
        self.speed = cmd.drive.speed
        self.steering_angle = cmd.drive.steering_angle
        self.timer_callback()  # Check safety conditions

    def print_warning(self, warning: str) -> None:
        """To avoid spamming the console."""
        now = self.get_clock().now()
        time_since_warning = delta_time(self.last_debug_timestamp, now)
        if time_since_warning > 1.0:
            self.get_logger().warning(warning)
            self.last_debug_timestamp = now

    def timer_callback(self) -> None:
        now = self.get_clock().now()
        should_stop = False

        # Check watchdogs
        if self.watchdog_localize_topic:
            time_since_localize = delta_time(self.localize_timestamp, now)
            if time_since_localize > self.watchdog_period:
                self.print_warning(
                    f"Localization message not received for {time_since_localize:.3f} > {self.watchdog_period:.3f} s, stopping car"
                )
                should_stop = True
        time_since_scan = delta_time(self.scan_timestamp, now)
        if time_since_scan > self.watchdog_period:
            self.print_warning(
                f"Scan message not received for {time_since_scan:.3f} > {self.watchdog_period:.3f} s, stopping car"
            )
            should_stop = True

        # Check if we're too close to an obstacle in front
        close_in_front = self.front_distance < self.min_front_distance
        if close_in_front:
            self.print_warning(
                f"Front too close ({self.front_distance:.3f} m < {self.min_front_distance:.3f} m), stopping car"
            )
            should_stop = True

        # Check if we might crash in front
        will_crash = self.front_distance / (self.speed + 1e-6) < self.stopping_time
        if will_crash:
            self.print_warning(
                f"Front crash within {self.stopping_time:.3f} s ({self.front_distance:.3f} m at {self.speed:.3f} m/s), stopping car"
            )
            should_stop = True

        # Check if we might scrape a wall
        if self.steering_angle == 0:
            scrape_dist = max(self.left_distance, self.right_distance)
        elif self.steering_angle > 0:
            scrape_dist = self.left_distance  # Turning towards left wall
        else:
            scrape_dist = self.right_distance  # Turning towards right wall
        scrape = scrape_dist < self.min_side_distance
        if scrape:
            self.print_warning(
                f"Might scrape side wall ({self.steering_angle:.3f} rad, {scrape_dist:.3f} m < {self.min_side_distance:.3f} m), stopping car"
            )
            should_stop = True

        # Stop if any condition is met
        if should_stop:
            self.ackermann_drive_msg_mut.header.stamp = now.to_msg()
            self.safety_pub.publish(self.ackermann_drive_msg_mut)


def main(args=None):
    rclpy.init(args=args)

    safety_controller = SafetyController()

    rclpy.spin(safety_controller)

    safety_controller.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
