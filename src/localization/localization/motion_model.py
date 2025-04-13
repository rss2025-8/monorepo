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
        Updates the particles in place.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

            dt: Time since last odometry update

            deterministic: Whether to force determinism
        """
        # Add a base amount of noise to the odometry in m/s or rad/s
        # Also add noise proportional to linear/angular "velocity"
        # (Odometry is ~50 Hz in both sim and real)
        dx, dy, dtheta = odometry

        # 1.0 m/s base noise, 1/3 * dx added noise
        x_dev = 1.0 * dt + np.abs(dx) / 3
        # 0.5 m/s base noise, 1/3 * dy added noise
        y_dev = 0.5 * dt + np.abs(dy) / 3
        # pi/6 rad/s base noise, 1/2 * dtheta added noise
        theta_dev = np.pi / 6 * dt + np.abs(dtheta) / 2

        # Add noise if needed
        if not deterministic and not self.node.deterministic:
            noise = np.random.normal(loc=0, scale=(x_dev, y_dev, theta_dev), size=particles.shape)
            particles += noise

        # Apply 2D pose composition formulas directly
        cos_theta = np.cos(particles[:, 2])
        sin_theta = np.sin(particles[:, 2])
        particles[:, :2] += np.column_stack([cos_theta * dx - sin_theta * dy, sin_theta * dx + cos_theta * dy])
        # Faster arctan2 to make theta in [-pi, pi]
        particles[:, 2] = (particles[:, 2] + dtheta + np.pi) % (2 * np.pi) - np.pi
