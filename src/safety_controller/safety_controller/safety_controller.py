import math

import numpy as np
import rclpy
import rclpy.time
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
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
        ackermann_cmd_topic: str = (
            self.declare_parameter("ackermann_cmd_topic", "vesc/low_level/ackermann_cmd")
            .get_parameter_value()
            .string_value
        )
        safety_topic: str = (
            self.declare_parameter("safety_topic", "vesc/low_level/input/safety").get_parameter_value().string_value
        )

        self.stopping_time: float = self.declare_parameter("stopping_time", 1.0).get_parameter_value().double_value
        self.watchdog_period: float = self.declare_parameter("watchdog_period", 0.5).get_parameter_value().double_value

        self.scan_sub: Subscription = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.ackermann_sub: Subscription = self.create_subscription(
            AckermannDriveStamped, ackermann_cmd_topic, self.ackermann_cmd_callback, 10
        )
        self.safety_pub: Publisher = self.create_publisher(AckermannDriveStamped, safety_topic, 10)

        self.timer: Timer = self.create_timer(0.01, self.timer_callback)

        self.ackermann_drive_msg_mut: AckermannDriveStamped = AckermannDriveStamped()

        self.scan_timestamp = self.get_clock().now()
        self.ackermann_timestamp = self.get_clock().now()
        self.front_distance = float("inf")
        self.speed = 0.0
        self.accel = 0.0

        self.get_logger().info("Safety Controller Initialized")

    def scan_callback(self, scan: LaserScan) -> None:
        self.scan_timestamp = Time.from_msg(scan.header.stamp)

        min_index = scan_index(scan, -math.pi / 12)
        max_index = scan_index(scan, math.pi / 12)

        self.front_distance = min(scan.ranges[min_index : max_index + 1])

    def ackermann_cmd_callback(self, cmd: AckermannDriveStamped) -> None:
        self.ackermann_timestamp = Time.from_msg(cmd.header.stamp)

        self.speed = cmd.drive.speed
        self.accel = cmd.drive.acceleration

    def timer_callback(self) -> None:
        now = self.get_clock().now()
        # watchdog_scan = delta_time(self.scan_timestamp, now) > self.watchdog_period
        # watchdog_ackermann = (
        #     delta_time(self.ackermann_timestamp, now) > self.watchdog_period
        # )
        will_crash = self.front_distance / (self.speed + 1e-6) < self.stopping_time

        if will_crash:
            self.ackermann_drive_msg_mut.header.stamp = now.to_msg()
            self.safety_pub.publish(self.ackermann_drive_msg_mut)

        # # if watchdog_scan:
        # #     self.get_logger().error(
        # #         f"Scan message was not received for {self.watchdog_period}s, safety controller stopping car."
        # #     )
        # # if watchdog_ackermann:
        # #     self.get_logger().error(
        # #         f"Ackermann drive message was not received for {self.watchdog_period}s, safety controller stopping car."
        # #     )
        if will_crash:
            self.get_logger().error(
                f"Minimum lidar distance / commanded speed < stopping time: {self.stopping_time}, safety controller stopping car."
            )


def main(args=None):
    rclpy.init(args=args)

    safety_controller = SafetyController()

    rclpy.spin(safety_controller)

    safety_controller.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
