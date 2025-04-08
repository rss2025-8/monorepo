import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from scan_simulator_2d import PyScanSimulator2D
from tf_transformations import euler_from_quaternion

# Try to change to just `from scan_simulator_2d import PyScanSimulator2D`
# if any error re: scan_simulator_2d occurs


np.set_printoptions(threshold=sys.maxsize)


def fast_round(x: np.ndarray) -> np.ndarray:
    """Fast version of np.round(x) for non-negative x."""
    return (x + 0.5).astype(int)


class SensorModel:

    def __init__(self, node: Node):
        self.node = node
        node.declare_parameter("map_topic", "default")
        node.declare_parameter("num_beams_per_particle", 1)
        node.declare_parameter("scan_theta_discretization", 1.0)
        node.declare_parameter("scan_field_of_view", 1.0)
        node.declare_parameter("lidar_scale_to_map_scale", 1.0)

        self.map_topic = node.get_parameter("map_topic").get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter("num_beams_per_particle").get_parameter_value().integer_value
        self.scan_theta_discretization = (
            node.get_parameter("scan_theta_discretization").get_parameter_value().double_value
        )
        self.scan_field_of_view = node.get_parameter("scan_field_of_view").get_parameter_value().double_value
        self.lidar_scale_to_map_scale = (
            node.get_parameter("lidar_scale_to_map_scale").get_parameter_value().double_value
        )

        ####################################
        # Adjust these parameters
        # self.alpha_hit = 0.74
        # self.alpha_short = 0.07
        # self.alpha_max = 0.07
        # self.alpha_rand = 0.12
        # self.sigma_hit = 8.0

        self.alpha_hit = 0.73
        self.alpha_short = 0.1
        self.alpha_max = 0.06
        self.alpha_rand = 0.11
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        self.node.get_logger().info("Map topic: %s" % self.map_topic)
        self.node.get_logger().info("# beams per particle: %s" % self.num_beams_per_particle)
        self.node.get_logger().info("Scan theta discretization: %s" % self.scan_theta_discretization)
        self.node.get_logger().info("Scan field of view: %s" % self.scan_field_of_view)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization,
        )

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(OccupancyGrid, self.map_topic, self.map_callback, 1)
        self.gen_plot_k = None  # 10
        # self.gen_plot_k = 10

        # Plot sensor model distribution
        # D = 150
        # Z = range(self.table_width)
        # self.sensor_model_table_plot = self.sensor_model_table[Z, D]
        # fig, ax = plt.subplots()
        # ax.plot(Z, self.sensor_model_table_plot)
        # ax.set_xlabel("Measured Distance (px)")
        # ax.set_ylabel(f"P(Measured Distance | True Distance = {D} px)")
        # ax.set_title("Sensor Model Distribution")
        # plt.tight_layout()
        # plt.show()

        if self.gen_plot_k:
            # Plot probability distribution
            fig, ax = plt.subplots()
            self.particle_probs = np.ones(self.gen_plot_k) / self.gen_plot_k

            # Cumulative probability
            (self.line,) = ax.plot([], [])
            self.line.set_data(range(self.gen_plot_k), self.particle_probs)

            # self.bars = ax.bar(range(self.gen_plot_k), [0] * self.gen_plot_k)
            # for bar, val in zip(self.bars, self.particle_probs):
            #     bar.set_height(val)
            ax.set_xlim(0, self.gen_plot_k - 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Particle Index")
            ax.set_ylabel("Probability")
            ax.set_title("Sensor Model Particle Probabilities")

            plt.ion()  # Turn on interactive mode
            plt.tight_layout()
            plt.show()

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

    def evaluate(self, particles: np.ndarray, raw_observation: list):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            raw_observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            # self.node.get_logger().warning("MAP NOT SET, SENSOR MODEL NOT EVALUATING")
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle

        # Downsample LIDAR scans
        # self.node.get_logger().info(f"Downsampling from {len(raw_observation)} to {self.num_beams_per_particle} beams")
        indices = fast_round(np.linspace(0, len(raw_observation) - 1, self.num_beams_per_particle))
        # self.node.get_logger().info(f"Indices: {indices}")
        observation = np.array(raw_observation)[indices]
        # self.node.get_logger().info(f"observation: {max(observation)}")

        # Get "ground truth" scans from each particle, knowing the lidar is slightly in front of the robot's center
        dist_in_front = 0.25
        # Move [dx, dy, dtheta] in the robot's frame of reference for each particle
        # lidar_offset = (
        #     np.stack([np.cos(particles[:, 2]), np.sin(particles[:, 2]), np.zeros(len(particles))], axis=1)
        #     * dist_in_front
        # )
        # scans = self.scan_sim.scan(particles + lidar_offset)
        scans = self.scan_sim.scan(particles)

        # Convert to pixels
        z_max = self.table_width - 1
        scans_pixels = scans / (self.resolution * self.lidar_scale_to_map_scale)
        scans_pixels = fast_round(np.clip(scans_pixels, 0, z_max))  # N x K
        observation_pixels = observation / (self.resolution * self.lidar_scale_to_map_scale)
        observation_pixels = fast_round(np.clip(observation_pixels, 0, z_max))  # K

        # scan_probs[i, j] = P(measured z_j pixels | d_j pixels at particle pose i)
        scan_probs = self.sensor_model_table[observation_pixels[None, :], scans_pixels]  # N x K

        # particle_probs[i] = P(particle i has all valid scans)
        particle_probs = np.prod(scan_probs, axis=1)  # N

        # (P1 * P2 * P3 * ... * Pn)^(1/n)
        # log(P1 * P2 * P3 * ... * Pn)^(1/n) = (1/n) * (log(P1) + log(P2) + log(P3) + ... + log(Pn))
        scaling_factor = 1 / self.num_beams_per_particle
        # Unsquash probabilities
        scaling_factor *= 50
        # scaling_factor *= 10
        # scaling_factor *= self.num_beams_per_particle
        # Decrease to squash more
        particle_probs = np.exp(np.log(particle_probs) * scaling_factor)

        # TODO fine tune / adjust probability distribution
        # Normalize probabilities
        particle_probs = particle_probs / np.sum(particle_probs)

        if self.gen_plot_k:
            self.particle_probs = particle_probs[np.argsort(particle_probs)[::-1]][: self.gen_plot_k]
            self.line.set_data(range(len(self.particle_probs)), self.particle_probs)
            # for bar, val in zip(self.bars, self.particle_probs):
            #     bar.set_height(val)
            plt.ylim(0, 1.0)
            plt.tight_layout()
            plt.pause(0.01)

        return particle_probs

        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.0
        self.map = np.clip(self.map, 0, 1)

        self.resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = euler_from_quaternion((origin_o.x, origin_o.y, origin_o.z, origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map, map_msg.info.height, map_msg.info.width, map_msg.info.resolution, origin, 0.5
        )  # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        self.node.get_logger().info("Map and sensor model initialized!")
