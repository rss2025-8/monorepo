import numpy as np
import rclpy
import tf2_ros
import tf_transformations
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseArray,
    PoseWithCovarianceStamped,
    Quaternion,
    TransformStamped,
    Vector3,
)
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, String, Float32

from localization.motion_model import MotionModel
from localization.sensor_model import SensorModel

assert rclpy


def pose_to_vec(msg: Pose) -> np.ndarray:
    """
    Pose message -> [x, y, yaw] ccw+
    ^ more useful docs than ros tf_transformations opencv and drake combined
    """

    quaternion = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]

    _, _, yaw = tf_transformations.euler_from_quaternion(quaternion)

    return np.array([msg.position.x, msg.position.y, yaw])


def point_to_pose(x, y, theta) -> Pose:
    quaternion = tf_transformations.quaternion_about_axis(theta, (0, 0, 1))
    return Pose(
        position=Point(x=x, y=y, z=0.0),
        orientation=Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]),
    )


def pose_to_tf(pose: Pose, parent: str, child: str, time: rclpy.time.Time) -> TransformStamped:
    """Converts a rotation matrix to a TransformStamped message.

    Trailing parameters follow the same format as TF's lookup_transform().
    """
    header = Header(stamp=time.to_msg(), frame_id=parent)
    msg = TransformStamped(header=header, child_frame_id=child)
    # Setup rotation and translation portions of transform
    msg.transform.rotation = Quaternion(
        x=pose.orientation.x, y=pose.orientation.y, z=pose.orientation.z, w=pose.orientation.w
    )
    msg.transform.translation = Vector3(x=pose.position.x, y=pose.position.y, z=pose.position.z)
    return msg


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

        self.debug = True  # cursed

        self.declare_parameter("particle_filter_frame", "default")
        self.particle_filter_frame = self.get_parameter("particle_filter_frame").get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        #
        # ^ ros best practices are dumb

        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("scan_topic", "/scan")

        scan_topic: str = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic: str = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.num_particles: int = self.declare_parameter("num_particles", 200).get_parameter_value().integer_value
        self.flip_odometry: bool = self.declare_parameter("flip_odometry", False).get_parameter_value().bool_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic, self.laser_callback, 1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.pose_callback, 1)

        self.odom_avg_mut = Odometry(child_frame_id=self.particle_filter_frame, header=Header(frame_id="/map"))
        # self.odom_avg_mut..

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)
        self.pose_error_pub = self.create_publisher(Float32, "/pf/pose/error", 1)

        self.debug_particles_pub = self.create_publisher(PoseArray, "/debug_particles", 1)

        self.debug_particle_array_mut = PoseArray(header=Header(frame_id="/map"))

        # Initialize the models
        # OK this is *cursed*
        self.motion_model = MotionModel(self)
        self.get_logger().info("noise level: " + str(self.motion_model.noise_level))
        self.sensor_model = SensorModel(self)
        self.gt_pose = None # For graphing the difference in calculated vs ground truth pose
        self.initial_pose_msg = None
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

        self.particles: np.ndarray = np.zeros((self.num_particles, 3))

        self.transform_broadcaster = tf2_ros.TransformBroadcaster(self)  # Broadcast TF transforms

        """
        particles: An Nx3 matrix of the form:

            [x0 y0 theta0]
            [x1 y0 theta1]
            [    ...     ]
        """

        self.prev_time: Time = self.get_clock().now()

    def pose_callback(self, initial_pose: PoseWithCovarianceStamped) -> None:
        self.gt_pose = pose_to_vec(initial_pose.pose.pose)
        self.particles = np.tile(pose_to_vec(initial_pose.pose.pose), (self.num_particles, 1)) + np.random.normal(
            scale=0.1, size=(self.num_particles, 3)
        )
    def odom_callback(self, odom: Odometry) -> None:
        """
        Run prediction step
        """
        new_time = Time.from_msg(odom.header.stamp)
        dt = (new_time - self.prev_time).nanoseconds / 1e9
        self.prev_time = new_time

        lin_vel = odom.twist.twist.linear
        rot_vel = odom.twist.twist.angular

        # TODO add intentional drift to odometry
        # lin_vel.x += np.random.normal(loc=0, scale=0.1)
        # lin_vel.y += np.random.normal(loc=0, scale=0.1)
        # rot_vel.z += np.random.normal(loc=0, scale=np.pi / 12)

        delta_pose = np.array([lin_vel.x * dt, lin_vel.y * dt, rot_vel.z * dt])
        
        if self.flip_odometry:
            delta_pose *= -1
        if self.gt_pose is not None:
            theta = self.gt_pose[2]
            dx_world = lin_vel.x * np.cos(theta) - lin_vel.y * np.sin(theta)
            dy_world = lin_vel.x * np.sin(theta) + lin_vel.y * np.cos(theta)
            delta_pose = np.array([dx_world * dt, dy_world * dt, rot_vel.z * dt])
            self.gt_pose += delta_pose
        self.get_logger().info(f"odometry: {delta_pose.round(4)}")

        self.particles = self.motion_model.evaluate(self.particles, delta_pose)
        self.publish_average_pose()

    def laser_callback(self, scan: LaserScan) -> None:
        # TODO move particles a bit with motion model to align with scan time (better accuracy)
        # ~1,081 ranges, [-2.356194496154785, 2.356194496154785]
        weights = self.sensor_model.evaluate(self.particles, scan.ranges)
        if weights is None:
            self.get_logger().warning("SENSOR MODEL NOT UPDATING, RELAUNCH MAP")
            return
        if self.debug:
            sorted_weights = weights[np.argsort(weights)][::-1].round(3)
            self.get_logger().info(f"sensor: {sorted_weights[:3]}...")

        # Resample
        indices = np.random.choice(self.particles.shape[0], self.num_particles, p=weights)
        self.particles = self.particles[indices]
        self.publish_average_pose()

    def publish_average_pose(self) -> None:
        # TODO maybe publish pose every 100 Hz or something (using rough odometry estimate in between updates)
        # Would allow much more consistent and fine-grained pose updates for later use
        # TODO fix (see better notion of average note in ipynb)
        x, y = np.mean(self.particles[:, :2], axis=0)

        angles = self.particles[:, 2]
        sigma_sin = np.sum(np.sin(angles))
        sigma_cos = np.sum(np.cos(angles))
        theta = np.arctan2(sigma_sin, sigma_cos)

        self.odom_avg_mut.pose.pose = point_to_pose(x, y, theta)
        self.odom_avg_mut.header.stamp = self.get_clock().now().to_msg()
        self.odom_pub.publish(self.odom_avg_mut)

        map_to_base_link = pose_to_tf(
            self.odom_avg_mut.pose.pose, "map", self.particle_filter_frame, self.get_clock().now()
        )
        self.transform_broadcaster.sendTransform(map_to_base_link)

        if self.debug:
            self.debug_particle_array_mut.poses = [point_to_pose(x, y, theta) for x, y, theta in self.particles]
            self.debug_particles_pub.publish(self.debug_particle_array_mut)
        if self.gt_pose is not None:
            estimated_pose = np.array([x, y, theta])
            error = np.linalg.norm(self.gt_pose[:-1] - estimated_pose[:-1])
            self.get_logger().info(f"ground truth: {self.gt_pose[:-1]}")
            self.get_logger().info(f"estimated: {estimated_pose[:-1]}")
            msg = Float32(data=error)
            self.pose_error_pub.publish(msg)
            self.get_logger().info(f"error: {error:.4f}")

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
