"""
Wall follower that does not fit a line (instead, it just uses raw LIDAR points).
"""

#!/usr/bin/env python3
import math
from typing import Optional

import numpy as np
import rclpy
import rclpy.time
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from numpy.polynomial import Polynomial
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from wall_follower.visualization_tools import VisualizationTools


class PID:
    """A Proportional-Integral-Derivative controller.

    The accumulated error (for Ki) is not clipped or reset.
    """

    def __init__(self, time: rclpy.time.Time, kp: float, ki=0.0, kd=0.0):
        """Creates a PID controller with the given initial time, Kp, Ki, and Kd."""
        self.prev_time = time
        self.kp, self.ki, self.kd = kp, ki, kd
        self.prev_error = 0.0
        self.total_error = 0.0  # Integral of error w.r.t. t

    def update(self, error: float, time: rclpy.time.Time) -> float:
        """Update PID state with the current error and the time that error was *measured*. Returns a new control output.

        For example, with Kp = 2 and Ki = Kd = 0, an error of 1 results in a control output of -2.
        """
        dt = (time.nanoseconds - self.prev_time.nanoseconds) / 1e9  # Time elapsed in seconds
        d_error = (error - self.prev_error) / dt  # d(Error)/dt
        self.total_error += error / dt
        correction = -(self.kp * error + self.kd * d_error + self.ki * self.total_error)
        self.prev_time, self.prev_error = time, error
        return correction


class WallFollower(Node):
    """Follows a wall on the given side. 1 = Left wall, -1 = Right wall."""

    def __init__(self):
        """Creates a WallFollower node."""
        super().__init__("wall_follower")

        # Declare parameters with defaults to make them available for use
        self.declare_parameter("scan_topic", "default")
        self.declare_parameter("drive_topic", "default")
        self.declare_parameter("side", 1)
        self.declare_parameter("velocity", 1.0)
        self.declare_parameter("desired_distance", 1.0)

        # Make PID parameters
        self.declare_parameter("Kp", 2.7)
        self.declare_parameter("Ki", 0.0)
        self.declare_parameter("Kd", 0.0)

        # Line follower parameters
        # How much to divide the front wall's distance by (to make it appear closer)
        self.declare_parameter("front_dist_weight", 2.0)
        # Additional values that can be tuned:
        # Angle ranges for side / front walls (side_wall_min/max, front_wall_min/max)
        self.side_wall_min, self.side_wall_max = math.radians(-15), math.radians(90)
        # Maximum distance of a point from the car for it to be considered (max_base_dist)
        self.max_base_dist = 4.0

        # Amount of debug info to print
        self.declare_parameter("debug", 1)

        # Fetch constants from the ROS parameter server
        # This is necessary for the tests to be able to test varying parameters!
        self.SCAN_TOPIC = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter("drive_topic").get_parameter_value().string_value
        self.SIDE = self.get_parameter("side").get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter("velocity").get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter("desired_distance").get_parameter_value().double_value
        # This activates the parameters_callback function so the tests can change parameters.
        self.add_on_set_parameters_callback(self.parameters_callback)

        # Init PID
        kp = self.get_parameter("Kp").get_parameter_value().double_value
        ki = self.get_parameter("Ki").get_parameter_value().double_value
        kd = self.get_parameter("Kd").get_parameter_value().double_value
        self.pid = PID(self.get_clock().now(), kp, ki, kd)

        self.FRONT_DIST_WEIGHT = self.get_parameter("front_dist_weight").get_parameter_value().double_value

        # For debugging (launch file params only update with colcon build)
        self.DEBUG = self.get_parameter("debug").get_parameter_value().integer_value
        # self.SIDE = 1  # Left wall
        # self.VELOCITY = 2.0
        # self.DESIRED_DISTANCE = 0.6

        # Init publishers and subscribers
        self.laser_sub = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.on_laser_scan, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)
        self.line_pub = self.create_publisher(Marker, "/line", 1)  # Side wall
        self.line_2_pub = self.create_publisher(Marker, "/line_2", 1)  # Front wall

    def fit_line(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Robustly fits a line to the given (x, y) coordinates. Returns (m, b) such that y ~ mx + b.

        Handles vertical lines by slanting it a tiny bit."""
        assert len(x) == len(y) and len(x) > 1
        try:
            b, m = Polynomial.fit(x, y, 1).convert().coef
        except Exception as e:  # Assume near-vertical line, slant it a little bit
            self.get_logger().warning(f"{type(e).__name__}: {str(e)} while trying to fit a line")
            x[0] += 1e-6
            x[-1] -= 1e-6
            b, m = Polynomial.fit(x, y, 1).convert().coef
        return m, b

    def drive(self, steering_angle: float, speed: float):
        """Publishes to the drive topic with the given steering angle (radians) and speed (m/s)."""
        header = Header(stamp=self.get_clock().now().to_msg())
        drive = AckermannDrive(
            steering_angle=steering_angle,
            steering_angle_velocity=0.0,
            speed=speed,
            acceleration=0.0,
            jerk=0.0,
        )
        self.drive_pub.publish(AckermannDriveStamped(header=header, drive=drive))

    def find_wall(self, ranges: np.ndarray, angles: np.ndarray, wall_angle_min: float, wall_angle_max: float):
        """Finds the line equation of a wall in [wall_angle_min, wall_angle_max] radians, where 0 is in front.

        Returns (m, b, X, Y) such that y ~ mx + b. X and Y are the points in the wall. Accounts for the side parameter.
        """
        # Filter ranges to those in the wall
        assert wall_angle_min < wall_angle_max
        wall_indices = (angles * self.SIDE >= wall_angle_min) & (angles * self.SIDE <= wall_angle_max)
        relevant_ranges = ranges.copy()[wall_indices]
        relevant_angles = angles.copy()[wall_indices]

        # When fitting a line, only consider points < max_base_dist + desired_distance away from the car
        # Filter ranges that are too far away, but fallback to all points if none are close enough
        close_indices = relevant_ranges < self.max_base_dist + self.DESIRED_DISTANCE
        if np.count_nonzero(close_indices) >= 2:
            relevant_ranges = relevant_ranges[close_indices]
            relevant_angles = relevant_angles[close_indices]

        # Fit line to the wall
        X = relevant_ranges * np.cos(relevant_angles)
        Y = relevant_ranges * np.sin(relevant_angles)
        m, b = self.fit_line(X, Y)
        return m, b, X, Y

    def on_laser_scan(self, msg: LaserScan):
        """Called on a new laser scan result. Updates the wall follower's drive command."""
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        # Weight point distances
        for i, angle in enumerate(angles):
            # Weight points in front as closer to the car (up to X times closer)
            if abs(angle) <= -self.side_wall_min:
                ranges[i] /= self.FRONT_DIST_WEIGHT - (self.FRONT_DIST_WEIGHT - 1) * abs(angle) / self.side_wall_max

        # Find wall on the side
        side_m, side_b, side_X, side_Y = self.find_wall(ranges, angles, self.side_wall_min, self.side_wall_max)
        # side_dist = abs(side_b) / np.sqrt(1 + side_m**2)
        # Only use the points for now
        side_dist = min(np.hypot(side_X, side_Y))

        # Weighted distance to the nearest wall
        weighted_dist = side_dist

        # Update PID and send drive command
        error = weighted_dist - self.DESIRED_DISTANCE
        steering_angle = self.pid.update(error, rclpy.time.Time.from_msg(msg.header.stamp)) * -self.SIDE
        speed = self.VELOCITY
        self.drive(steering_angle, speed)

        if self.DEBUG >= 1:
            # Visualize the line
            # side_X = np.array([-1.0, 1.0])
            # side_Y = side_m * side_X + side_b
            VisualizationTools.plot_line(side_X, side_Y, self.line_pub, color=(1.0, 0.0, 0.0), frame="/laser")
            # VisualizationTools.plot_line(front_X, front_Y, self.line_2_pub, color=(0.0, 1.0, 1.0), frame="/laser")
        if self.DEBUG >= 2:
            # Debug info
            self.get_logger().info(
                # f"Wall: y = {m:.2f}x + {b:.2f}, "
                f"Dist: {weighted_dist:.2f} m, "
                f"Angle: {steering_angle:5.2f} rad = {math.degrees(steering_angle):3.0f}°, "
                f"{speed:.2f} m/s"
            )

    def parameters_callback(self, params):
        """
        DO NOT MODIFY THIS CALLBACK FUNCTION!

        This is used by the test cases to modify the parameters during testing.
        It's called whenever a parameter is set via 'ros2 param set'.
        """
        for param in params:
            if param.name == "side":
                self.SIDE = param.value
                self.get_logger().info(f"Updated side to {self.SIDE}")
            elif param.name == "velocity":
                self.VELOCITY = param.value
                self.get_logger().info(f"Updated velocity to {self.VELOCITY}")
            elif param.name == "desired_distance":
                self.DESIRED_DISTANCE = param.value
                self.get_logger().info(f"Updated desired_distance to {self.DESIRED_DISTANCE}")
        return SetParametersResult(successful=True)


def main():
    rclpy.init()
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
