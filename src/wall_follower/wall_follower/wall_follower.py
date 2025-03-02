"""
Wall follower.

Uses P control (no derivative or integral term), with a trick to make points
in front of the robot appear closer than they are when calculating error.

Visualization: https://gyazo.com/adef8c4daa3d48515ab404f052312272
"""

#!/usr/bin/env python3
import math

import numpy as np
import rclpy
import rclpy.time
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from wall_follower.visualization_tools import VisualizationTools


class PID:
    """Represents a Proportional-Integral-Derivative controller."""

    K_P = 2.7
    K_I = 0.0
    K_D = 0.0

    def __init__(self, time: rclpy.time.Time):
        """Creates a PID controller with the given initial time."""
        self.time = time
        self.last_error = 0  # TODO jarring start

    def update(self, error: float, time: rclpy.time.Time) -> float:
        """Update with the current error and the time it was measured. Returns a new control output."""
        duration: rclpy.time.Duration = time - self.time
        d_error = (error - self.last_error) / (duration.nanoseconds / 1e9)  # d(Error)/dt
        correction = -error * self.K_P - d_error * self.K_D
        self.time = time
        self.last_error = error
        return correction


class WallFollower(Node):
    """Follows a wall on the given side. 1 = Left, -1 = Right."""

    def __init__(self):
        """Creates a WallFollower node."""
        super().__init__("wall_follower")
        # Declare parameters to make them available for use
        # DO NOT MODIFY THIS!
        self.declare_parameter("scan_topic", "default")
        self.declare_parameter("drive_topic", "default")
        self.declare_parameter("side", 1)
        self.declare_parameter("velocity", 1.0)
        self.declare_parameter("desired_distance", 1.0)

        # Fetch constants from the ROS parameter server
        # DO NOT MODIFY THIS! This is necessary for the tests to be able to test varying parameters!
        self.SCAN_TOPIC = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter("drive_topic").get_parameter_value().string_value
        self.SIDE = self.get_parameter("side").get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter("velocity").get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter("desired_distance").get_parameter_value().double_value

        # This activates the parameters_callback function so that the tests are able
        # to change the parameters during testing.
        # DO NOT MODIFY THIS!
        self.add_on_set_parameters_callback(self.parameters_callback)

        # # For debugging
        # self.SIDE = 1
        # self.VELOCITY = 3.0

        # Init publishers and subscribers
        self.laser_sub = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.on_laser_scan, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)
        self.filtered_pub = self.create_publisher(LaserScan, "/filtered_scan", 10)
        self.line_pub = self.create_publisher(Marker, "/closest_line", 1)

        # Init PID
        self.pid = PID(self.get_clock().now())

    def create_drive_message(self, steering_angle: float, speed: float) -> AckermannDriveStamped:
        """Creates an AckermannDriveStamped message with the specified steering_angle (radians) and speed (m/s)."""
        # Setup and publish AckermannDriveStamped message
        header = Header(stamp=self.get_clock().now().to_msg())
        drive = AckermannDrive(
            steering_angle=steering_angle,
            steering_angle_velocity=0.0,
            speed=speed,
            acceleration=0.0,
            jerk=0.0,
        )
        # Setup messages and ranges
        return AckermannDriveStamped(header=header, drive=drive)

    def on_laser_scan(self, msg: LaserScan):
        """Finds and publishes the longest return (angle/distance)."""
        # TODO Convert code to numpy
        # Angles of LIDAR points to consider
        ANGLE_LIMIT_WALL_SIDE = math.radians(90)  # Include more point on the wall side
        ANGLE_LIMIT_OPEN_SIDE = math.radians(15)  # Include a bit of points on the non-wall side (corners)
        FRONT_WEIGHT = 2.5  # Amount to shrink distance of points in front by
        points = []  # Filtered to only (x, y) points on the wall
        filtered_ranges = []  # For visualization only
        for i, dist in enumerate(msg.ranges):
            angle = msg.angle_min + (msg.angle_increment * i)
            if -ANGLE_LIMIT_OPEN_SIDE <= angle * self.SIDE <= ANGLE_LIMIT_WALL_SIDE:
                # Weight points in front as closer to the car (up to X times closer)
                if abs(angle) <= ANGLE_LIMIT_OPEN_SIDE:
                    dist /= FRONT_WEIGHT - (FRONT_WEIGHT - 1) * abs(angle) / ANGLE_LIMIT_OPEN_SIDE
                points.append((dist * math.cos(angle), dist * math.sin(angle)))
                filtered_ranges.append(dist)
            else:
                filtered_ranges.append(float("inf"))

        # Calculate error and steering angle
        min_dist = min([math.hypot(x, y) for x, y in points])
        error = min_dist - self.DESIRED_DISTANCE
        steering_angle = self.pid.update(error, rclpy.time.Time.from_msg(msg.header.stamp)) * -self.SIDE
        speed = self.VELOCITY

        # Debug info
        # self.get_logger().info(
        #     # f"{len(msg.ranges)} -> {len(points)} points, " +
        #     f"Wall dist: {min_dist:4.3f} m, "
        #     f"Angle: {steering_angle:5.2f} rad =  {math.degrees(steering_angle):3.0f}°, "
        #     # f"{speed:.2f} m/s"
        # )

        # Publish filtered LaserScan data
        filtered_msg = msg
        filtered_msg.header.stamp = self.get_clock().now().to_msg()
        filtered_msg.ranges = filtered_ranges
        self.filtered_pub.publish(filtered_msg)

        # Publish closest point line
        # TODO Could be cleaner
        closest_point = points[0]
        for point in points:
            if math.hypot(*point) < math.hypot(*closest_point):
                closest_point = point
        VisualizationTools.plot_line(
            (0.0, closest_point[0]), (0.0, closest_point[1]), self.line_pub, color=(0.0, 1.0, 1.0), frame="/laser"
        )

        # Drive
        self.drive_pub.publish(self.create_drive_message(steering_angle, speed))

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
