#!/usr/bin/env python

import math
from enum import Enum, auto

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from rclpy.node import Node
from std_msgs.msg import Header
from visual_servoing.visualization_tools import VisualizationTools
from visualization_msgs.msg import Marker
from vs_msgs.msg import ConeLocation, ParkingError


class State(Enum):
    IDLE = auto()
    TOO_CLOSE = auto()
    NORMAL = auto()


class PID:
    """
    A Proportional-Integral-Derivative controller.
    The accumulated error (for Ki) is not clipped or reset.
    """

    def __init__(self, time: rclpy.time.Time, kp: float, ki=0.0, kd=0.0):
        """Creates a PID controller with the given initial time, Kp, Ki, and Kd."""
        self.prev_time = time
        self.kp, self.ki, self.kd = kp, ki, kd
        self.prev_error = 0.0
        self.total_error = 0.0  # Integral of error w.r.t. t

    def update(self, error: float, time: rclpy.time.Time) -> float:
        """
        Update PID state with the current error and the time that error was *measured*. Returns a new control output.

        For example, with Kp = 2 and Ki = Kd = 0, an error of 1 results in a control output of -2.
        """
        dt = (time.nanoseconds - self.prev_time.nanoseconds) / 1e9  # Time elapsed in seconds
        d_error = (error - self.prev_error) / dt  # d(Error)/dt
        self.total_error += error / dt
        correction = -(self.kp * error + self.kd * d_error + self.ki * self.total_error)
        self.prev_time, self.prev_error = time, error
        return correction


class ParkingController(Node):
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """

    def __init__(self):
        super().__init__("parking_controller")

        #################################

        # Set in launch file; different for simulator vs racecar
        self.declare_parameter("drive_topic", "default")

        # Parking controller parameters
        self.declare_parameter("parking_distance", 0.75 * 3)  # Desired parking distance (meters); play with this!
        self.declare_parameter("max_distance_error", 0.03)  # Acceptable distance error (meters)
        self.declare_parameter("max_angle_error", 2.0)  # Acceptable angle error (degrees)
        self.declare_parameter("max_velocity", 1.0)  # Max velocity (m/s)
        self.declare_parameter("min_velocity", 0.01)  # Min velocity that allows the car to move (m/s)
        self.declare_parameter("x_point_turn_length", 0.5)  # Margin above parking_distance to perform turns in (meters)

        # Parking distance PID
        self.declare_parameter("Dp", 10.0)
        self.declare_parameter("Di", 0.0)
        self.declare_parameter("Dd", 0.2)

        # Angle error PID
        self.declare_parameter("Ap", 2.0)
        self.declare_parameter("Ai", 0.0)
        self.declare_parameter("Ad", 0.1)

        #################################

        # Retrieve parameter values
        DRIVE_TOPIC = self.get_parameter("drive_topic").value

        self.parking_distance = self.get_parameter("parking_distance").value
        self.max_distance_error = self.get_parameter("max_distance_error").value
        self.max_angle_error = self.get_parameter("max_angle_error").value
        self.max_angle_error = math.radians(self.max_angle_error)
        self.max_velocity = self.get_parameter("max_velocity").value
        self.min_velocity = self.get_parameter("min_velocity").value
        self.x_point_turn_length = self.get_parameter("x_point_turn_length").value

        Dp = self.get_parameter("Dp").value
        Di = self.get_parameter("Di").value
        Dd = self.get_parameter("Dd").value
        Ap = self.get_parameter("Ap").value
        Ai = self.get_parameter("Ai").value
        Ad = self.get_parameter("Ad").value

        # Initialize node
        self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)
        self.line_drive_pub = self.create_publisher(Marker, "/line_drive", 1)
        self.text_state_pub = self.create_publisher(Marker, "/text_state", 1)

        self.create_subscription(ConeLocation, "/relative_cone", self.relative_cone_callback, 1)

        # Publish drive commands at a minimal rate (20 Hz)
        self.drive_cmd = AckermannDrive()
        self.drive_timer = self.create_timer(1 / 20, self.timer_callback)

        self.state = State.IDLE
        self.relative_x = 0
        self.relative_y = 0

        self.pid_distance = PID(self.get_clock().now(), Dp, Di, Dd)
        self.pid_angle = PID(self.get_clock().now(), Ap, Ai, Ad)

        self.get_logger().info("Parking Controller Initialized")

    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        msg_time = self.get_clock().now()

        #################################

        # Find relative distance and angle
        relative_distance = math.hypot(self.relative_x, self.relative_y)
        relative_angle = math.atan2(self.relative_y, self.relative_x)

        # State transitions
        prev_state = self.state
        distance_error = relative_distance - self.parking_distance
        if abs(distance_error) <= self.max_distance_error and abs(relative_angle) <= self.max_angle_error:
            self.state = State.IDLE  # Close enough to park
        elif distance_error <= self.max_distance_error:
            self.state = State.TOO_CLOSE  # Need to turn while getting away from the cone
        elif self.state == State.TOO_CLOSE:
            # Give enough time to turn (so it doesn't oscillate between too close and normal)
            if (
                abs(relative_angle) <= self.max_angle_error
                or distance_error > self.max_distance_error + self.x_point_turn_length
            ):
                self.state = State.NORMAL  # Far enough from the cone (or aligned)
        else:
            self.state = State.NORMAL  # Far enough from the cone

        if prev_state != self.state:
            self.get_logger().info(
                f"{prev_state.name} -> {self.state.name}"
                f", Cone: {relative_distance:.2f} m, {math.degrees(relative_angle):.1f}°"
            )
        elif self.state == State.IDLE:
            self.get_logger().info(f"Parked. Cone: {relative_distance:.2f} m, {math.degrees(relative_angle):.1f}°")

        # Drive command
        if self.state == State.IDLE:
            # Stop the car
            steering_angle, velocity = 0.0, 0.0
        elif self.state == State.NORMAL:
            # Angle should face the cone, velocity should decrease as we near the parking spot
            if relative_angle < -math.pi * 0.95:
                relative_angle = math.pi  # Ensure the car doesn't get stuck oscillating left/right
            steering_angle = -self.pid_angle.update(relative_angle, msg_time)
            velocity = -self.pid_distance.update(distance_error, msg_time)
        elif self.state == State.TOO_CLOSE:
            # Get away from the cone
            if abs(relative_angle) > math.pi * 2 / 3:  # math.pi / 2 can cause an infinite circling situation
                # Go forwards
                velocity = self.max_velocity
            else:
                # Go backwards
                velocity = -self.max_velocity
            steering_angle = -self.pid_angle.update(relative_angle, msg_time)

        #################################

        velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
        if velocity != 0.0 and abs(velocity) < self.min_velocity:
            velocity = velocity * self.min_velocity / abs(velocity)  # So the car always moves when it's not idle
        steering_angle = np.clip(steering_angle, -math.pi / 2, math.pi / 2)  # So it doesn't wrap around
        if velocity < 0.0:
            steering_angle *= -1  # Reverse steering should still turn the car toward the cone

        # Visualize for debugging
        if velocity != 0.0:
            visual_drive_X = [0.0, velocity / self.max_velocity * math.cos(steering_angle) / 2]
            visual_drive_Y = [0.0, velocity / self.max_velocity * math.sin(steering_angle) / 2]
        else:
            visual_drive_X, visual_drive_Y = [], []
        brightness = 0.5 if abs(velocity) == self.max_velocity else 0.0
        VisualizationTools.plot_line(
            visual_drive_X, visual_drive_Y, self.line_drive_pub, color=(brightness, 1.0, brightness), frame="/laser"
        )
        VisualizationTools.plot_text(self.state.name, 0.0, 0.0, 0.5, 0.2, self.text_state_pub, frame="/laser")

        self.drive(steering_angle, velocity)
        self.error_publisher()

    def error_publisher(self):
        """
        Publish the error between the car and the cone (ideal parking position has 0 error).
        We will view this with rqt_plot to plot the success of the controller.
        """
        error_msg = ParkingError()
        error_msg.x_error = self.relative_x - self.parking_distance
        error_msg.y_error = self.relative_y
        error_msg.distance_error = math.hypot(self.relative_x - self.parking_distance, self.relative_y)
        self.error_pub.publish(error_msg)

    def timer_callback(self):
        """Ensure drive commands are sent at a regular rate."""
        self.drive(use_last_cmd=True)

    def drive(self, steering_angle=0.0, velocity=0.0, use_last_cmd=False):
        """
        Publishes to the drive topic with the given steering angle (radians) and velocity (m/s).
        If `use_last_cmd` is set, uses the last drive command.
        """
        header = Header(stamp=self.get_clock().now().to_msg())
        if not use_last_cmd:
            self.drive_cmd = AckermannDrive(
                steering_angle=steering_angle,
                steering_angle_velocity=0.0,
                speed=velocity,
                acceleration=0.0,
                jerk=0.0,
            )
        self.drive_pub.publish(AckermannDriveStamped(header=header, drive=self.drive_cmd))


def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
