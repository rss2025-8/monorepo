import math
import threading

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
from std_msgs.msg import Float32, Header, String

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
    """Convert (x, y, theta) to a Pose message."""
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
        self.prev_time: Time = self.get_clock().now()
        self.prev_odom: Odometry = Odometry()

        self.debug: bool = self.declare_parameter("debug", False).value
        self.num_particles: int = self.declare_parameter("num_particles", 200).value
        self.forward_offset: float = self.declare_parameter("forward_offset", 0.0).value
        self.deterministic: bool = self.declare_parameter("deterministic", False).value
        self.on_racecar: bool = self.declare_parameter("on_racecar", False).value
        self.particle_filter_frame: str = self.declare_parameter("particle_filter_frame", "default").value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        #
        # ^ ros best practices are dumb
        scan_topic: str = self.declare_parameter("scan_topic", "/scan").value
        odom_topic: str = self.declare_parameter("odom_topic", "/odom").value
        self.laser_sub = self.create_subscription(LaserScan, scan_topic, self.laser_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.pose_callback, 1)
        self.odom_avg_mut = Odometry(child_frame_id=self.particle_filter_frame, header=Header(frame_id="/map"))

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.
        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)
        self.pose_error_pub = self.create_publisher(Float32, "/pf/pose/error", 1)

        self.debug_particles_pub = self.create_publisher(PoseArray, "/debug_particles", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)
        self.true_pose = None
        self.initial_pose_msg = None

        self.get_logger().info(f"# particles: {self.num_particles}")
        self.get_logger().info(
            "# beams per particle | normalized: "
            f"{self.sensor_model.num_beams_per_particle} | {self.sensor_model.normalized_beams}"
        )

        if self.debug:
            self.get_logger().warning("NOTE: Debug mode enabled, expect slow performance!")
            self.get_logger().warning("Debug particle publisher only gives the first ~100 particles.")
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
        self.particles: np.ndarray = np.zeros((self.num_particles, 3))  # Nx3 matrix of [xi yi thetai]
        self.transform_broadcaster = tf2_ros.TransformBroadcaster(self)  # Broadcast TF transforms
        self.sensor_model_working = True  # To prevent spamming logs

    def pose_callback(self, initial_pose: PoseWithCovarianceStamped) -> None:
        initial_vec = pose_to_vec(initial_pose.pose.pose)
        # Experiments: Add intentional drift to initial pose
        # initial_vec += np.random.normal(loc=0, scale=(1, 1, np.pi / 16), size=3)
        self.particles = np.tile(initial_vec, (self.num_particles, 1)) + np.random.normal(
            loc=0, scale=(0.3, 0.3, np.pi / 6), size=(self.num_particles, 3)
        )

    def odom_callback(self, odom: Odometry) -> None:
        """Run prediction step. Should run consistently at 50 Hz."""
        call_time = self.get_clock().now()
        new_time = Time.from_msg(odom.header.stamp)
        dt = (new_time - self.prev_time).nanoseconds / 1e9
        self.prev_time = new_time
        self.prev_odom = odom
        # self.get_logger().info(f"odom dt:  {dt:.4f}")
        self.true_pose = pose_to_vec(odom.pose.pose)

        lin_vel = odom.twist.twist.linear
        rot_vel = odom.twist.twist.angular
        # Experiments: Add intentional drift to odometry
        # lin_vel.x += np.random.normal(loc=0, scale=2.0)
        # lin_vel.y += np.random.normal(loc=0, scale=2.0)
        # rot_vel.z += np.random.normal(loc=0, scale=np.pi / 3)
        vel = np.array([lin_vel.x, lin_vel.y, rot_vel.z])
        delta_pose = (-1 if self.on_racecar else 1) * dt * vel
        # self.get_logger().info(f"odometry: {delta_pose.round(4)}")

        self.motion_model.evaluate(self.particles, delta_pose, dt)
        self.publish_average_pose(new_time)

        latency = (self.get_clock().now() - call_time).nanoseconds / 1e9
        if latency > 0.015:
            self.get_logger().info(f"high odom callback latency, check debug or num_particles: {latency:.4f}s")

    def laser_callback(self, scan: LaserScan) -> None:
        """Run update step. Runs at ~40 Hz (car), ~50 Hz (sim).

        Assumes this function consistently takes <1/50 of a second to run."""
        call_time = self.get_clock().now()
        new_time = Time.from_msg(scan.header.stamp)
        dt = (new_time - self.prev_time).nanoseconds / 1e9
        if dt > 0.01:
            # Move particles with motion model / last known odometry to align with scan time
            # self.get_logger().info(f"high laser dt: {dt:.4f}")
            lin_vel = self.prev_odom.twist.twist.linear
            rot_vel = self.prev_odom.twist.twist.angular
            prev_vel = np.array([lin_vel.x, lin_vel.y, rot_vel.z])
            delta_pose = (-1 if self.on_racecar else 1) * dt * prev_vel
            moved_particles = self.particles.copy()
            self.motion_model.evaluate(moved_particles, delta_pose, dt, deterministic=True)
        else:
            moved_particles = self.particles

        # Find particle weights
        weights = self.sensor_model.evaluate(moved_particles, scan)
        # weights = self.sensor_model.evaluate(self.particles, scan)
        if weights is None:
            if self.sensor_model_working:
                self.get_logger().warning("SENSOR MODEL NOT UPDATING, RELAUNCH MAP")
                self.sensor_model_working = False
            return
        if not self.sensor_model_working:
            self.sensor_model_working = True
        # sorted_weights = weights[np.argsort(weights)][::-1].round(3)
        # self.get_logger().info(f"sensor: {sorted_weights[:3]}...")

        # Resample
        indices = np.random.choice(self.particles.shape[0], self.num_particles, p=weights)
        self.particles = self.particles[indices]

        latency = (self.get_clock().now() - call_time).nanoseconds / 1e9
        if latency > 0.015:
            self.get_logger().info(f"high laser callback latency, check debug or num_particles: {latency:.4f}s")

    def publish_average_pose(self, time: Time) -> None:
        """Publish the average pose of the particles at the given time.

        Publishes only on every odometry update (~50 Hz), for consistency when being used later.
        Could be changed.
        """

        # Get average x, y, and theta (circular mean)
        x, y = np.mean(self.particles[:, :2], axis=0)
        angles = self.particles[:, 2]
        sigma_sin = np.sum(np.sin(angles))
        sigma_cos = np.sum(np.cos(angles))
        theta = np.arctan2(sigma_sin, sigma_cos)

        # Move (x, y) to account for lidar being slightly in front of the robot
        x += self.forward_offset * math.cos(theta)
        y += self.forward_offset * math.sin(theta)

        # Publish average pose and TF transform
        self.odom_avg_mut.pose.pose = point_to_pose(x, y, theta)
        self.odom_avg_mut.header.stamp = time.to_msg()
        self.odom_pub.publish(self.odom_avg_mut)
        map_to_base_link = pose_to_tf(self.odom_avg_mut.pose.pose, "map", self.particle_filter_frame, time)
        self.transform_broadcaster.sendTransform(map_to_base_link)

        # Find error in sim
        if not self.on_racecar and self.true_pose is not None:
            truth_pose = self.true_pose
            estimated_pose = [x, y, theta]
            error = np.linalg.norm(truth_pose[:2] - estimated_pose[:2])
            msg = Float32(data=error)
            self.pose_error_pub.publish(msg)

        if self.debug:
            # Publish some of the particles, but not all (very slow)
            num_to_publish = min(len(self.particles), 100)
            debug_msg = PoseArray(header=Header(frame_id="/map"))
            debug_msg.poses = [
                point_to_pose(
                    x + self.forward_offset * math.cos(theta), y + self.forward_offset * math.sin(theta), theta
                )
                for x, y, theta in self.particles[:num_to_publish]
            ]
            self.debug_particles_pub.publish(debug_msg)


# For profiling
# import atexit
# import cProfile

# profiler = cProfile.Profile()


# def save_profile():
#     profiler.disable()
#     profiler.dump_stats("profile.prof")
#     print("Saved profile to profile.prof")


# atexit.register(save_profile)


def main(args=None):
    # profiler.enable()

    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
    rclpy.shutdown()
