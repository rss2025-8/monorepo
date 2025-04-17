#!/usr/bin/env python3
import math
import time

import rclpy
import tf_transformations
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseArray,
    PoseWithCovariance,
    PoseWithCovarianceStamped,
    Quaternion,
)
from rclpy.node import Node
from std_msgs.msg import Header

from path_planning.utils import LineTrajectory


def point_to_pose(x, y, theta) -> Pose:
    """Convert (x, y, theta) to a Pose message."""
    quaternion = tf_transformations.quaternion_about_axis(theta, (0, 0, 1))
    return Pose(
        position=Point(x=x, y=y, z=0.0),
        orientation=Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]),
    )


class LoadTrajectory(Node):
    """Loads a trajectory from the file system and publishes it to a ROS topic."""

    def __init__(self):
        super().__init__("trajectory_loader")

        self.declare_parameter("trajectory", "default")
        self.path = self.get_parameter("trajectory").get_parameter_value().string_value

        # publish initial pose estimate
        self.initial_pose_topic = "/initialpose"
        self.pose_sub = self.create_publisher(PoseWithCovarianceStamped, self.initial_pose_topic, 1)

        # initialize and load the trajectory
        self.trajectory = LineTrajectory(self, "/planned_trajectory")
        self.get_logger().info(f"Loading from {self.path}")
        self.trajectory.load(self.path)

        self.pub_topic = "/trajectory/current"
        self.traj_pub = self.create_publisher(PoseArray, self.pub_topic, 1)

        # need to wait a short period of time before publishing the first message
        time.sleep(0.5)

        # publish initial pose estimate
        # Trajectory 1 basic case
        x, y, theta = -9.663803740201708, -0.5357283449568534, math.pi * 6 / 7
        pose_without_covariance = point_to_pose(x, y, theta)
        pose_with_covariance = PoseWithCovarianceStamped(
            header=Header(stamp=self.get_clock().now().to_msg(), frame_id="map"),
            pose=PoseWithCovariance(pose=pose_without_covariance, covariance=[0.0 for _ in range(36)]),
        )
        self.pose_sub.publish(pose_with_covariance)

        # visualize the loaded trajectory
        # self.trajectory.publish_viz()

        # send the trajectory
        self.publish_trajectory()

    def publish_trajectory(self):
        print("Publishing trajectory to:", self.pub_topic)
        self.traj_pub.publish(self.trajectory.toPoseArray())


def main(args=None):
    rclpy.init(args=args)
    load_trajectory = LoadTrajectory()
    rclpy.spin(load_trajectory)
    load_trajectory.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
