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
        result = []

        # adding noise, change the bounds (these are just arbitrary and have no meaning)
        if not self.deterministic:
            odometry[0] += np.random.normal(-1, -1) # x-value noise
            odometry[1] += np.random.normal(-1, -1) # y-value noise
            odometry[2] += np.random.norma(-math.pi, math.pi) #theta noise

        odom_matrix = np.array([
        [math.cos(odometry[2]), -math.sin(odometry[2]), odometry[0]],
        [math.sin(odometry[2]),  math.cos(odometry[2]), odometry[1]],
        [0,                      0,                     1]])

        for particle in particles:
            particle = np.array([
            [math.cos(particle[2]), -math.sin(particle[2]), particle[0]],
            [math.sin(particle[2]),  math.cos(particle[2]), particle[1]],
            [0,                      0,                     1]])
            pose = (particle@odom_matrix).tolist()
            # self.node.get_logger().info(f"{pose}")
            result.append([pose[0][2], pose[1][2], math.atan2(pose[1][0],pose[0][0])])
        # self.node.get_logger().info(f"result: {result}")
        return np.array(result)
