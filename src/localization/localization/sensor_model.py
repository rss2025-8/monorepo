import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import range_libc

    fallback = False
except ImportError:
    from scan_simulator_2d import PyScanSimulator2D
    from tf_transformations import euler_from_quaternion

    fallback = True

from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker

from localization.visualization_tools import VisualizationTools

# np.set_printoptions(threshold=sys.maxsize)


class SensorModel:

    def __init__(self, node: Node):
        self.node = node

        self.map_topic: str = node.declare_parameter("map_topic", "default").value
        self.num_beams_per_particle: int = node.declare_parameter("num_beams_per_particle", 1).value
        self.normalized_beams: int = node.declare_parameter("normalized_beams", 10).value
        self.scan_theta_discretization: float = node.declare_parameter("scan_theta_discretization", 1.0).value
        self.scan_field_of_view: float = node.declare_parameter("scan_field_of_view", 1.0).value
        self.lidar_scale_to_map_scale: float = node.declare_parameter("lidar_scale_to_map_scale", 1.0).value

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

        # Cached variables
        self.scans = np.empty(self.node.num_particles * self.num_beams_per_particle, dtype=np.float32)  # NK, cached
        self.indices = None
        self.angles = None

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(OccupancyGrid, self.map_topic, self.map_callback, 1)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        self.debug_raytracing_pub = node.create_publisher(Marker, "/debug_raytracing", 1)

        if fallback:
            # Create a simulated laser scan
            self.scan_sim = PyScanSimulator2D(
                self.num_beams_per_particle,
                self.scan_field_of_view,
                0,  # This is not the simulator, don't add noise
                0.01,  # This is used as an epsilon
                self.scan_theta_discretization,
            )

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
        # Iterate over all possible z_k and d_k values
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

    def evaluate(self, particles: np.ndarray, scan: LaserScan):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            scan: The actual LIDAR scan

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """
        if not self.map_set:
            return  # Need a map to evaluate sensor model
        assert particles.shape[0] == self.node.num_particles

        if self.indices is None:
            # Precompute relevant indices and angles
            self.indices = np.rint(np.linspace(0, len(scan.ranges) - 1, self.num_beams_per_particle)).astype(int)
            self.angles = np.array(self.indices, dtype=np.float32) * scan.angle_increment + scan.angle_min
            # self.node.get_logger().info(f"indices: {indices[:5]}")
            # self.node.get_logger().info(f"angles: {angles[:5]}")

        # Downsample LIDAR scans
        raw_observation = scan.ranges
        observation = np.array(raw_observation, dtype=np.float32)[self.indices]

        if not fallback:
            # Get ground truth with raytracing
            particle_probs = np.ones(particles.shape[0], dtype=np.float64)  # N
            self.range_method.calc_range_repeat_angles(particles.astype(np.float32), self.angles, self.scans)
            self.range_method.eval_sensor_model(
                observation, self.scans, particle_probs, self.num_beams_per_particle, particles.shape[0]
            )
        else:
            # Get ground truth with raytracing
            scans = self.scan_sim.scan(particles)  # N x K
            # Convert to pixels
            z_max = self.table_width - 1
            scans_pixels = scans / (self.resolution * self.lidar_scale_to_map_scale)
            scans_pixels = np.rint(np.clip(scans_pixels, 0, z_max)).astype(int)  # N x K
            observation_pixels = observation / (self.resolution * self.lidar_scale_to_map_scale)
            observation_pixels = np.rint(np.clip(observation_pixels, 0, z_max)).astype(int)  # K
            # scan_probs[i, j] = P(measured z_j pixels | d_j pixels at particle pose i)
            scan_probs = self.sensor_model_table[observation_pixels[None, :], scans_pixels]  # N x K
            # particle_probs[i] = P(particle i has all valid scans)
            particle_probs = np.prod(scan_probs, axis=1)  # N

        # Visualize the raytracing
        if self.node.debug:
            if fallback:
                self.scans = scans.flatten()
            # Get points from the raytracing particle with highest probability
            best_idx = np.argmax(particle_probs)
            # best_particle = particles[best_idx]
            # Get points from the raytracing
            best_scans = self.scans[best_idx * self.num_beams_per_particle + np.arange(self.num_beams_per_particle)]
            # best_scans = observation
            X = (best_scans * np.cos(self.angles)).tolist()
            Y = (best_scans * np.sin(self.angles)).tolist()
            # Visualize the points
            VisualizationTools.plot_points(X, Y, self.debug_raytracing_pub)

        # Squash probabilities to be equivalent to having K beams of information
        particle_probs **= self.normalized_beams / self.num_beams_per_particle

        # Normalize probabilities
        particle_probs = particle_probs / np.sum(particle_probs)
        return particle_probs

    def map_callback(self, map_msg):
        self.node.get_logger().info("Initializing map and sensor model...")

        # Add a 1px wall (value 100) around edges of map (of type array.array) for range_libc
        for i in range(map_msg.info.width):
            map_msg.data[i] = 100
            map_msg.data[i + map_msg.info.width * (map_msg.info.height - 1)] = 100
        for i in range(map_msg.info.height):
            map_msg.data[i * map_msg.info.width] = 100
            map_msg.data[i * map_msg.info.width + map_msg.info.width - 1] = 100

        # Convert the map to a numpy array (-1 -> 0, 0 -> 0, 100 -> 1)
        self.map = np.array(map_msg.data, np.double) / 100.0
        self.map = np.clip(self.map, 0, 1)
        self.resolution = map_msg.info.resolution
        self.map_set = True

        if not fallback:
            # Setup range_libc
            if self.node.use_gpu:
                self.node.get_logger().info("Using racecar range method (ray marching + GPU)...")
                oMap = range_libc.PyOMap(map_msg)
                self.range_method = range_libc.PyRayMarchingGPU(oMap, self.table_width - 1)
            else:
                self.node.get_logger().info("Using local range method (ray marching)...")
                oMap = range_libc.PyOMap(map_msg)
                self.range_method = range_libc.PyRayMarching(oMap, self.table_width - 1)
            self.range_method.set_sensor_model(self.sensor_model_table)
        else:
            self.node.get_logger().warning("***range_libc module not found, falling back to PyScanSimulator***")
            # Convert the origin to a tuple
            origin_p = map_msg.info.origin.position
            origin_o = map_msg.info.origin.orientation
            origin_o = euler_from_quaternion((origin_o.x, origin_o.y, origin_o.z, origin_o.w))
            origin = (origin_p.x, origin_p.y, origin_o[2])

            # Initialize a map with the laser scan
            self.scan_sim.set_map(
                self.map, map_msg.info.height, map_msg.info.width, map_msg.info.resolution, origin, 0.5
            )  # Consider anything < 0.5 to be free

        self.node.get_logger().info("Map and sensor model initialized!")
