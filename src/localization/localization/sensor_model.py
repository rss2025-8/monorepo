import numpy as np
from scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs

from tf_transformations import euler_from_quaternion

from nav_msgs.msg import OccupancyGrid

import sys

np.set_printoptions(threshold=sys.maxsize)


class SensorModel:

    def __init__(self, node):
        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', 1)
        node.declare_parameter('scan_theta_discretization', 1.0)
        node.declare_parameter('scan_field_of_view', 1.0)
        node.declare_parameter('lidar_scale_to_map_scale', 1.0)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

        ####################################
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)
    
    def p_hit(self, z, d, z_max):
        if 0 <= z <= z_max:
            return (1 / np.sqrt(2 * np.pi * self.sigma_hit**2)) * np.exp(-0.5 * ((z - d) / self.sigma_hit) ** 2)
        return 0

    def p_short(self, z, d):
        if 0 <= z <= d and d != 0:
            return (2 / d) * (1 - (z / d))
        return 0

    def p_max(self, z, z_max):
        if z == z_max:
            return 1
        return 0

    def p_rand(self, z, z_max):
        if 0 <= z <= z_max:
            return 1 / z_max
        return 0

    def probability_without_hit(self, z, d, z_max):
        p_short = self.alpha_short * self.p_short(z, d)
        p_max = self.alpha_max * self.p_max(z, z_max)
        p_rand = self.alpha_rand * self.p_rand(z, z_max)
        return p_short + p_max + p_rand

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        # iterate over all possible z_k and d_k values
        # d should be columns and z should be rows
        p_hit_result = np.zeros((self.table_width, self.table_width))
        no_hit_result = np.zeros((self.table_width, self.table_width))
        for z in range(self.table_width):
            for d in range(self.table_width):
                no_hit_result[z, d] = self.probability_without_hit(z, d, self.table_width - 1)
                p_hit_result[z, d] = self.p_hit(z, d, self.table_width - 1)
        # Normalize across columns for p_hit_result
        p_hit_result = p_hit_result / np.sum(p_hit_result, axis=0, keepdims=True)
        # add in the p hit into the no hit result
        self.sensor_model_table = (self.alpha_hit * p_hit_result) + no_hit_result
        # Normalize across columns
        self.sensor_model_table = self.sensor_model_table / np.sum(self.sensor_model_table, axis=0, keepdims=True)

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 

        # TODO downsample

        scans = self.scan_sim.scan(particles)

        # TODO could optimize
        scans_pixels = scans / (self.resolution * self.lidar_scale_to_map_scale)
        scans_pixels = (np.clip(scans_pixels, 0, self.table_width - 1) + 0.5).astype(int)  # N x K
        measured_pixels = observation / (self.resolution * self.lidar_scale_to_map_scale)
        measured_pixels = (np.clip(measured_pixels, 0, self.table_width - 1) + 0.5).astype(int)  # N

        # scan_probs[i][j] = P(measured z_j pixels | d_j pixels at particle pose i)
        scan_probs = self.sensor_model_table[measured_pixels[None, :], scans_pixels]  # N x K

        # particle_probs[i] = P(particle i has all valid scans)
        particle_probs = np.prod(scan_probs, axis=1)  # N
        return particle_probs

        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.clip(self.map, 0, 1)

        self.resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = euler_from_quaternion((
            origin_o.x,
            origin_o.y,
            origin_o.z,
            origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)  # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Graph a 3d plot of the sensor model table to verify correctness no ros
