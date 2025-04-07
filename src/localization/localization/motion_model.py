import math

import numpy as np
from rclpy.node import Node


class MotionModel:

    def __init__(self, node: Node):
        ####################################
        # Do any precomputation for the motion
        # model here.

        self.node = node
        # change this to False when not deterministic/adding noise
        node.declare_parameter("deterministic", False)
        self.deterministic = node.get_parameter("deterministic").get_parameter_value().bool_value
        self.last_time = node.get_clock().now()
        self.noise_level = 0.3
        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """
        ####################################

        # Add noise in a way such that laggy odometry (small vs. large dt) doesn't change noise "amount"
        # Noise increases when linear and/or angular velocity is high
        # Sends every ~50 Hz in both sim and real

        curr_time = self.node.get_clock().now()
        dt = (curr_time - self.last_time).nanoseconds / 1e9
        self.last_time = curr_time

        vx = odometry[0] / dt
        vy = odometry[1] / dt
        vtheta = odometry[2] / dt

        # 1.0 m/s base noise, 1/2 * velocity added noise
        x_dev = (1.0 + 0.5 * np.abs(vx)) * dt
        y_dev = (1.0 + 0.5 * np.abs(vy)) * dt
        # pi/4 rad/s base noise, 1 * vtheta added noise
        theta_dev = (np.pi / 4 + 1.0 * np.abs(vtheta)) * dt

        # On the car (old): noise_level = 0.08, np.pi / 70

        # Add noise if needed (odometry is ~50 Hz)
        # self.noise_level = 0.3
        if not self.deterministic:
            # noise = np.random.normal(
            #     loc=0, scale=(self.noise_level, self.noise_level, np.pi / 100), size=particles.shape
            # )
            noise = np.random.normal(loc=0, scale=(x_dev, y_dev, theta_dev), size=particles.shape)
            particles += noise

        dx, dy, dtheta = odometry
        cos_theta = np.cos(particles[:, 2])
        sin_theta = np.sin(particles[:, 2])

        # Apply 2D pose composition formulas directly
        new_x = particles[:, 0] + cos_theta * dx - sin_theta * dy
        new_y = particles[:, 1] + sin_theta * dx + cos_theta * dy
        new_theta = particles[:, 2] + dtheta
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi  # Faster arctan2 to make theta in [-pi, pi]
        return np.column_stack((new_x, new_y, new_theta))
