import numpy as np

from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from std_msgs.msg import String, Header
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from sensor_msgs.msg import LaserScan

from rclpy.node import Node
import rclpy
from rclpy.time import Time

import tf_transformations

assert rclpy


def pose_to_vec(msg: Pose) -> np.ndarray:
    """
        Pose message -> [x, y, yaw] ccw+
        ^ more useful docs than ros tf_transformations opencv and drake combined
    """

    quaternion = [
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
        msg.orientation.w
    ]

    _, _, yaw = tf_transformations.euler_from_quaternion(quaternion)

    return np.array([msg.position.x, msg.position.y, yaw])


def point_to_pose(x, y, theta) -> Pose:
    quaternion = tf_transformations.quaternion_about_axis(theta, (0, 0, 1))
    return Pose(
        position=Point(
            x=x,
            y=y,
            z=0
        ),
        orientation=Quaternion(
            x=quaternion[0],
            y=quaternion[1],
            z=quaternion[2],
            w=quaternion[3]
        )
    )


class ParticleFilter(Node):

    def __init__(self):
        """
            Prediction:
                run motion model to update all particles to next timestamp
            Update:
                re-assign weights to updated particles based on measurement model
            Resampling:
                prune low probability particles
                i think duplicate highly weighed particles?
            is this even supposed to be a callback?
        """
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        #
        # ^ ros best practices are dumb

        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic: str = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic: str = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.num_particles: int = self.declare_parameter("num_particles", 200).get_parameter_value().integer_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        self.odom_avg_mut = Odometry(
            child_frame_id=self.particle_filter_frame,
            header=Header(frame_id="/map")
        )
        # self.odom_avg_mut..

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize the models
        # OK this is *cursed*
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.
        #

        self.particles: np.ndarray = np.empty((self.num_particles, 3))
        """
        particles: An Nx3 matrix of the form:

            [x0 y0 theta0]
            [x1 y0 theta1]
            [    ...     ]
        """

        self.prev_time: Time = self.get_clock().now()

    def pose_callback(self, initial_pose: PoseWithCovarianceStamped) -> None:
        self.particles = np.tile(pose_to_vec(initial_pose.pose.pose), (self.num_particles, 3))

    def odom_callback(self, odom: Odometry) -> None:
        """
            Run prediction step
        """
        new_time = Time.from_msg(odom.header.stamp)
        dt = (new_time - self.prev_time).nanoseconds
        self.prev_time = new_time

        lin_vel = odom.twist.twist.linear
        rot_vel = odom.twist.twist.angular
        delta_pose = np.array([lin_vel.x * dt, lin_vel.y * dt, rot_vel.z * dt])

        self.particles = self.motion_model.evaluate(self.particles, delta_pose)
        self.publish_average_pose()

    def laser_callback(self, scan: LaserScan) -> None:

        weights = self.sensor_model.evaluate(self.particles, scan.ranges)

        # resample (probably broken)
        self.particles = np.random.choice(self.particles, self.num_particles, p=weights)
        self.publish_average_pose()

    def publish_average_pose(self) -> None:
        # TODO fix
        x, y = np.mean(self.particles[:, :2], axis=0)

        angles = self.particles[:2]
        sigma_sin = sum(np.sin(angles))
        sigma_cos = sum(np.cos(angles))
        theta = np.arctan2(sigma_sin, sigma_cos)

        self.odom_avg_mut.pose.pose = point_to_pose(x, y, theta)        
        self.odom_avg_mut.header.stamp = self.get_clock().now().to_msg()


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
