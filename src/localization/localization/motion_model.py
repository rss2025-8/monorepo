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

        # TODO add noise in a way such that laggy odometry (small vs. large dt) doesn't change noise "amount"
        # Also noise should increase when linear and/or angular velocity is high

        # Add noise if needed (odometry is ~50 Hz)
        if not self.deterministic:
            noise = np.random.normal(loc=0, scale=(0.01, 0.01, np.pi / 300), size=particles.shape)
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
