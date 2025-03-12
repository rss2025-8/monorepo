"""
Wall follower that detects a side and front wall and responds accordingly.
The front wall just uses points instead of fitting a line for stability.
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
        self.declare_parameter("wall_follower_topic", "default")
        self.declare_parameter("side", 1)
        self.declare_parameter("velocity", 2.0)
        self.declare_parameter("desired_distance", 1.0)

        # Good for velocity 1 in simulation: Kp = 5.7, Ap = 4.0, all others = 0.0
        # Good for velocity 2 in simulation: Kp = 5.7, Ap = 2.5, Ad = 0.05, all others = 0.0
        # Good for velocity 1 on the racecar: Kp = 0.75, Kd = 0.1, Ap = 0.2, all others = 0.0

        # PID parameters (simulation by default)
        self.declare_parameter("Kp", 5.7)
        self.declare_parameter("Ki", 0.0)
        self.declare_parameter("Kd", 0.0)
        self.declare_parameter("Ap", 4.0)
        self.declare_parameter("Ai", 0.0)
        self.declare_parameter("Ad", 0.0)

        # Lidar filter parameters
        self.declare_parameter("side_wall_min", -30.0)  # Degrees
        self.declare_parameter("side_wall_max", 90.0)  # Degrees
        self.declare_parameter("max_dist_between_points", 0.6)  # Meters
        self.declare_parameter("weighted_dist_between_points", 2.5)

        # Amount of debug info to print
        self.declare_parameter("debug", 1)

        # Fetch constants from the ROS parameter server
        # This is necessary for the tests to be able to test varying parameters!
        self.SCAN_TOPIC = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter("wall_follower_topic").get_parameter_value().string_value
        self.SIDE = self.get_parameter("side").get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter("velocity").get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter("desired_distance").get_parameter_value().double_value
        # This activates the parameters_callback function so the tests can change parameters.
        self.add_on_set_parameters_callback(self.parameters_callback)

        # Init PIDs
        kp = self.get_parameter("Kp").get_parameter_value().double_value
        ki = self.get_parameter("Ki").get_parameter_value().double_value
        kd = self.get_parameter("Kd").get_parameter_value().double_value
        self.pid_dist = PID(self.get_clock().now(), kp, ki, kd)

        ap = self.get_parameter("Ap").get_parameter_value().double_value
        ai = self.get_parameter("Ai").get_parameter_value().double_value
        ad = self.get_parameter("Ad").get_parameter_value().double_value
        self.pid_angle = PID(self.get_clock().now(), ap, ai, ad)

        # For debugging (launch file params only update with colcon build)
        self.DEBUG = self.get_parameter("debug").get_parameter_value().integer_value
        # self.SIDE = 1  # Switch to left wall (default is right)
        # self.VELOCITY = 1.0

        # Filter values
        self.side_wall_min = math.radians(self.get_parameter("side_wall_min").get_parameter_value().double_value)
        self.side_wall_max = math.radians(self.get_parameter("side_wall_max").get_parameter_value().double_value)
        self.max_dist_between_points = self.get_parameter("max_dist_between_points").get_parameter_value().double_value
        self.weighted_dist_between_points = (
            self.get_parameter("weighted_dist_between_points").get_parameter_value().double_value
        )

        # Init publishers and subscribers
        self.laser_sub = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.on_laser_scan, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)
        self.line_pub = self.create_publisher(Marker, "/line", 1)
        self.line_2_pub = self.create_publisher(Marker, "/line_2", 1)
        self.line_3_pub = self.create_publisher(Marker, "/line_3", 1)
        self.line_4_pub = self.create_publisher(Marker, "/line_4", 1)

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

    def find_wall(
        self, ranges: np.ndarray, angles: np.ndarray, max_range: float, wall_angle_min: float, wall_angle_max: float
    ):
        """Finds the line equation of a wall in [wall_angle_min, wall_angle_max] radians, where 0 is in front.

        Returns (m, b, X, Y) such that y ~ mx + b. X and Y are the points in the wall. Accounts for the side parameter.
        """
        # Filter ranges to those in the wall
        assert wall_angle_min < wall_angle_max
        wall_indices = (angles * self.SIDE >= wall_angle_min) & (angles * self.SIDE <= wall_angle_max)
        relevant_ranges = ranges.copy()[wall_indices]
        relevant_angles = angles.copy()[wall_indices]

        # # When fitting a line, only consider points < max_base_dist + desired_distance away from the car
        # # Filter ranges that are too far away, but fallback to all points if none are close enough
        close_indices = relevant_ranges < max_range
        if np.count_nonzero(close_indices) >= 2:
            relevant_ranges = relevant_ranges[close_indices]
            relevant_angles = relevant_angles[close_indices]

        # Fit line to the wall
        X = relevant_ranges * np.cos(relevant_angles)
        Y = relevant_ranges * np.sin(relevant_angles)

        # Filter points with a large jump (no longer part of wall)
        if self.SIDE == 1:
            X, Y = X[::-1], Y[::-1]
        starting_i = max(2, len(relevant_angles) // 8)
        final_X = X[:starting_i].tolist()
        final_Y = Y[:starting_i].tolist()
        for i in range(starting_i, len(X)):
            dist = math.hypot(X[i] - final_X[-1], Y[i] - final_Y[-1])
            last_range = math.hypot(final_X[-1], final_Y[-1])
            # Manually tuned for the robot
            if (
                last_range > self.DESIRED_DISTANCE and dist > last_range / self.weighted_dist_between_points
            ) or dist > self.max_dist_between_points:  # Exclude points not on wall
                continue
                # break
            final_X.append(X[i])
            final_Y.append(Y[i])
        if self.SIDE == 1:
            final_X, final_Y = final_X[::-1], final_Y[::-1]

        m, b = self.fit_line(final_X, final_Y)
        return m, b, final_X, final_Y

    def on_laser_scan(self, msg: LaserScan):
        """Called on a new laser scan result. Updates the wall follower's drive command."""
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)

        # Find wall on the side
        side_m, side_b, side_X, side_Y = self.find_wall(
            ranges, angles, msg.range_max, self.side_wall_min, self.side_wall_max
        )
        side_dist = abs(side_b) / np.sqrt(1 + side_m**2)
        side_error = side_dist - self.DESIRED_DISTANCE
        angle_error = np.arctan(side_m)

        # Turn signal from side wall
        steering_dist = self.pid_dist.update(side_error, msg_time) * -self.SIDE
        # Turn signal from angle
        steering_angle = self.pid_angle.update(angle_error, msg_time) * -1

        # Combine turn signals, clip, and drive
        total_steering = steering_dist + steering_angle
        total_steering = min(max(total_steering, -math.pi / 2), math.pi / 2)
        speed = self.VELOCITY
        self.drive(total_steering, speed)

        steering_dist = min(max(steering_dist, -math.pi / 2), math.pi / 2)
        steering_angle = min(max(steering_angle, -math.pi / 2), math.pi / 2)

        if self.DEBUG >= 1:
            # Visualize turn angles
            steer_X, steer_Y = [0.0, 0.5 * math.cos(steering_dist)], [0.0, 0.5 * math.sin(steering_dist)]
            steer2_X, steer2_Y = [0.0, 0.5 * math.cos(steering_angle)], [0.0, 0.5 * math.sin(steering_angle)]
            steerall_X, steerall_Y = [0.0, 1 * math.cos(total_steering)], [0.0, 1 * math.sin(total_steering)]
            # Visualize the line
            # side_X = np.array([-1.0, 1.0])
            # side_Y = side_m * side_X + side_b
            # front_Y = np.array([-1.0, 1.0])
            # front_X = (front_Y - front_b) / front_m
            VisualizationTools.plot_line(side_X, side_Y, self.line_pub, color=(1.0, 0.0, 0.0), frame="/laser")
            VisualizationTools.plot_line(steer_X, steer_Y, self.line_2_pub, color=(1.0, 0.0, 0.0), frame="/laser")
            VisualizationTools.plot_line(steer2_X, steer2_Y, self.line_3_pub, color=(0.0, 1.0, 0.0), frame="/laser")
            VisualizationTools.plot_line(steerall_X, steerall_Y, self.line_4_pub, color=(0.0, 0.0, 1.0), frame="/laser")
        if self.DEBUG >= 2:
            # Debug info
            self.get_logger().info(
                # f"Wall: y = {m:.2f}x + {b:.2f}, "
                f"Dist: {side_dist:.2f} m, "
                f"Angle: {total_steering:5.2f} rad = {math.degrees(total_steering):3.0f}°, "
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
