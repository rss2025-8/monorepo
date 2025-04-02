import math
import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        # Do any precomputation for the motion
        # model here.

        # change this to False when not deterministic/adding noise
        self.deterministic = True
        self.node = node

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
        noise = 0

        cos_theta = math.cos(odometry[2])
        sin_theta = math.sin(odometry[2])

        odom_matrix = np.array([
        [cos_theta, -sin_theta, odometry[0]],
        [sin_theta,  cos_theta, odometry[1]],
        [0,          0,         1]])

        particles = np.array(particles)

        # adding noise, change the bounds (these are just arbitrary and have no meaning)
        if not self.deterministic:
            noise = np.random.normal(loc=[0, 0, 0], scale=[1, 1, math.pi], size=(len(particles), 3))
        particles += noise

        cos_theta_p = np.cos(particles[:, 2])
        sin_theta_p = np.sin(particles[:, 2])

        particle_matrices = np.zeros((len(particles), 3, 3))
        particle_matrices[:, 0, 0] = cos_theta_p
        particle_matrices[:, 0, 1] = -sin_theta_p
        particle_matrices[:, 0, 2] = particles[:, 0]
        particle_matrices[:, 1, 0] = sin_theta_p
        particle_matrices[:, 1, 1] = cos_theta_p
        particle_matrices[:, 1, 2] = particles[:, 1]
        particle_matrices[:, 2, 2] = 1

        transformed = np.einsum('ijk,kl->ijl', particle_matrices, odom_matrix)

        return np.column_stack((transformed[:, 0, 2], transformed[:, 1, 2], np.arctan2(transformed[:, 1, 0], transformed[:, 0, 0])))
