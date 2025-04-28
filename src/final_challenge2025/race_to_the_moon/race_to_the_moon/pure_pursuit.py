#!/usr/bin/env python

import numpy as np
import rclpy
import tf_transformations
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, Pose, Quaternion, TransformStamped, Vector3
from rclpy.node import Node
from std_msgs.msg import Header
from vs_msgs.msg import LookaheadLocation


class PurePursuit(Node):

    def __init__(self):
        super().__init__("pure_pursuit")

        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("look_ahead", 0.5)
        self.declare_parameter("wheelbase", 0.34)
        self.declare_parameter("default_velocity", 0.5)

        self.drive_topic = self.get_parameter("drive_topic").value
        self.look_ahead = self.get_parameter("look_ahead").value
        self.wheelbase = self.get_parameter("wheelbase").value
        self.velocity = self.get_parameter("default_velocity").value
        self.debug: bool = self.declare_parameter("debug", False).value

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.create_subscription(LookaheadLocation, "/relative_lookahead", self.relative_lookahead_callback, 1)

        self.relative_x = 0
        self.relative_y = 0

        if self.debug:
            self.get_logger().info("DEBUG mode enabled")
        self.get_logger().info("Pure pursuit initialized")

    def relative_lookahead_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        drive_cmd = AckermannDriveStamped()

        #################################

        # YOUR CODE HERE
        # Use relative position and your control law to set drive_cmd

        L = self.wheelbase
        L1 = self.look_ahead

        # Find the next reference point to pursue and calculate distance error and angle of reference point
        eta = np.arctan2(self.relative_y, self.relative_x)
        steer_angle = np.arctan2(2 * np.sin(eta) * L, L1)
        # PD control of speed
        # current_time = self.get_clock().now()
        # dt = (current_time.nanoseconds - self.prev_time.nanoseconds) * 1e-9
        # if dt <= 1e-6:
        #     dt = 1e-6

        # dist_err_derivative = (dist_err - self.prev_dist_err) / dt
        # speed_pd = self.Kp * dist_err + self.Kd * dist_err_derivative
        # speed_pd = max(0.0, min(speed_pd, 1.0))

        # self.prev_time = current_time
        # self.prev_dist_err = dist_err

        # If the robot is outside of 0.01 meters from the cone, move towards the cone with pure pursuit
        # Else if the robot is within 0.01 meters from the cone, stop and align with the cone

        drive_cmd.header.stamp = self.get_clock().now().to_msg()
        drive_cmd.header.frame_id = "base_link"
        drive_cmd.drive.steering_angle = steer_angle
        drive_cmd.drive.speed = self.velocity

        #################################

        self.drive_pub.publish(drive_cmd)


def main(args=None):
    rclpy.init(args=args)
    pure_pursuit = PurePursuit()
    rclpy.spin(pure_pursuit)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
