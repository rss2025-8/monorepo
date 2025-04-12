import math

import numpy as np
from rclpy.node import Node


class MotionModel:

    def __init__(self, node: Node):
        self.node = node

    def evaluate(self, particles: np.ndarray, odometry: np.ndarray, dt: float, deterministic: bool = None):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

            dt: Time since last odometry update

            deterministic: Whether to force determinism

        returns:
            particles: An updated matrix of the
                same size
        """
        # Add a base amount of noise to the odometry in m/s or rad/s
        # Also add noise proportional to linear/angular "velocity"
        # (Odometry is ~50 Hz in both sim and real)
        dx, dy, dtheta = odometry

        # # 6.0 m/s base noise, 1/2 * dx added noise
        # x_dev = 6.0 * dt + 0.5 * np.abs(dx)
        # # 2.0 m/s base noise, 1/2 * dy added noise
        # y_dev = 2.0 * dt + 0.5 * np.abs(dy)
        # # pi/3 rad/s base noise, 1/2 * dtheta added noise
        # theta_dev = np.pi / 3 * dt + 0.5 * np.abs(dtheta)

        # # 2.0 m/s base noise, 1/2 * dx added noise
        # x_dev = 2.0 * dt + 0.5 * np.abs(dx)
        # # 1.0 m/s base noise, 1/2 * dy added noise
        # y_dev = 1.0 * dt + 0.5 * np.abs(dy)
        # # pi/3 rad/s base noise, 1/2 * dtheta added noise
        # theta_dev = np.pi / 3 * dt + 0.5 * np.abs(dtheta)

        # 1.0 m/s base noise, 1/3 * dx added noise
        x_dev = 1.0 * dt + np.abs(dx) / 3
        # 0.5 m/s base noise, 1/3 * dy added noise
        y_dev = 0.5 * dt + np.abs(dy) / 3
        # pi/6 rad/s base noise, 1/2 * dtheta added noise
        theta_dev = np.pi / 6 * dt + np.abs(dtheta) / 2

        # # 0.5 m/s base noise, 1/4 * dx added noise
        # x_dev = 0.5 * dt + 0.25 * np.abs(dx)
        # # 0.25 m/s base noise, 1/4 * dy added noise
        # y_dev = 0.25 * dt + 0.25 * np.abs(dy)
        # # pi/8 rad/s base noise, 1/4 * dtheta added noise
        # theta_dev = np.pi / 8 * dt + 0.25 * np.abs(dtheta)

        # Add noise if needed
        if not deterministic and not self.node.deterministic:
            noise = np.random.normal(loc=0, scale=(x_dev, y_dev, theta_dev), size=particles.shape)
            particles += noise

        # Apply 2D pose composition formulas directly
        cos_theta = np.cos(particles[:, 2])
        sin_theta = np.sin(particles[:, 2])
        new_x = particles[:, 0] + cos_theta * dx - sin_theta * dy
        new_y = particles[:, 1] + sin_theta * dx + cos_theta * dy
        new_theta = particles[:, 2] + dtheta
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi  # Faster arctan2 to make theta in [-pi, pi]
        return np.column_stack((new_x, new_y, new_theta))
