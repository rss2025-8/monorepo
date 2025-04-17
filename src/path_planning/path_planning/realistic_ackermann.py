"""
More realistic Ackermann steering model. Only use this in simulation.
Takes AckermannDriveStamped messages and relays them in a more realistic way.
"""

import math
from typing import Optional

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from rclpy.node import Node
from std_msgs.msg import Header


class RealisticAckermann(Node):
    def __init__(self):
        super().__init__("realistic_ackermann")
        self.input_drive_topic: str = self.declare_parameter("input_drive_topic", "default").value
        self.output_drive_topic: str = self.declare_parameter("output_drive_topic", "default").value

        # True car dynamics
        self.k_slip = 0.08  # Experimentally determined
        self.drive_system_angle = SecondOrderSystem(omega_n=1.5, zeta=0.1)
        self.drive_system_velocity = SecondOrderSystem(omega_n=3.0, zeta=0.5)
        self.emulated_hz = float("inf")  # Drop enough commands to get around this Hz

        self.drive_time = 0
        self.input_drive_sub = self.create_subscription(
            AckermannDriveStamped, self.input_drive_topic, self.drive_callback, 1
        )
        self.output_drive_pub = self.create_publisher(AckermannDriveStamped, self.output_drive_topic, 1)
        self.get_logger().info(f"Realistic Ackermann initialized")

    def drive_callback(self, msg: AckermannDriveStamped):
        """Add realistic dynamics to the drive message."""
        steering_angle = msg.drive.steering_angle
        velocity = msg.drive.speed
        self.drive(steering_angle, velocity)

    def drive(self, steering_angle=0.0, velocity=0.0):
        """
        Publishes to the drive topic with the given steering angle (radians) and velocity (m/s).
        """
        if abs(steering_angle) > math.pi / 2:
            self.get_logger().warn(f"Steering angle {steering_angle} is too large")

        # [-0.34, 0.34 radians] on actual car
        steering_angle = np.clip(steering_angle, -math.pi / 2, math.pi / 2)
        max_velocity = 4.0
        velocity = np.clip(velocity, -max_velocity, max_velocity)

        dt = (self.get_clock().now().nanoseconds - self.drive_time) / 1e9

        # Tire slip model (higher speeds/angles = more slip)
        slip_effect = self.k_slip * velocity * np.tan(steering_angle)
        target_steering = steering_angle - slip_effect
        target_steering = np.clip(target_steering, -math.pi / 2, math.pi / 2)

        # Drop steering commands to emulate desired Hz
        if self.drive_system_angle.time_since_update < 1.0 / self.emulated_hz:
            target_steering = None
            velocity = None

        # Second order, more realistic
        new_steering_angle = self.drive_system_angle.update(dt, target_steering)
        new_steering_angle = np.clip(new_steering_angle, -math.pi / 2, math.pi / 2)
        new_velocity = self.drive_system_velocity.update(dt, velocity)
        new_velocity = np.clip(new_velocity, -max_velocity, max_velocity)

        # self.get_logger().info(f"Desired angle: {steering_angle}, Speed: {velocity}, dt: {dt}")
        # self.get_logger().info(f"Steering angle: {new_steering_angle}, Speed: {new_velocity}")
        if target_steering is None:
            return
        steering_angle = new_steering_angle
        velocity = new_velocity

        header = Header(stamp=self.get_clock().now().to_msg())
        self.drive_cmd = AckermannDrive(
            steering_angle=steering_angle,
            steering_angle_velocity=0.0,
            speed=velocity,
            acceleration=0.0,
            jerk=0.0,
        )
        self.output_drive_pub.publish(AckermannDriveStamped(header=header, drive=self.drive_cmd))
        self.drive_time = self.get_clock().now().nanoseconds


class SecondOrderSystem:
    """Models a second order system with a given natural frequency and damping ratio."""

    def __init__(self, omega_n: float, zeta: float, initial_value: float = 0.0, initial_velocity: float = 0.0):
        """Creates a model of asecond order system.

        Args:
            omega_n: Natural frequency (oscillation frequency with no damping).
            zeta: Damping ratio ("friction" of the system, 0 oscillates forever, <1 overshoots, >1 gradually returns).
            initial_value: Initial value of the system.
            initial_velocity: Initial velocity of the system.
        """
        self.omega_n = omega_n  # Natural frequency
        self.zeta = zeta  # Damping ratio
        self.value = initial_value
        self.velocity = initial_velocity
        self.setpoint = initial_value
        self.time_since_update = 0.0

    def update(self, dt: float, setpoint: Optional[float] = None) -> float:
        """
        Update the state of the system after a time step dt using the differential equation:
        a = omega_n^2 * (setpoint - value) - 2 * zeta * omega_n * v
        Euler integration is used to update the system. Returns the new value of the system.

        If setpoint is provided, it will be applied *after* dt seconds have elapsed.
        """
        # Acceleration from second order dynamics
        acceleration = (self.omega_n**2) * (self.setpoint - self.value) - 2 * self.zeta * self.omega_n * self.velocity
        # Update velocity and position using Euler integration
        self.velocity += acceleration * dt
        self.value += self.velocity * dt
        self.time_since_update += dt
        # New setpoint, applied after dt seconds
        if setpoint is not None:
            self.setpoint = setpoint
            self.time_since_update = 0.0
        return self.value


def main(args=None):
    rclpy.init(args=args)
    ackermann = RealisticAckermann()
    rclpy.spin(ackermann)
    rclpy.shutdown()
