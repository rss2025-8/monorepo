import math
from typing import Optional

import numpy as np
import rclpy
import tf_transformations
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from . import visualize
from .utils import LineTrajectory


class PurePursuit(Node):
    """Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed."""

    def __init__(self):
        super().__init__("trajectory_follower")
        self.odom_topic = self.declare_parameter("odom_topic", "default").value
        self.drive_topic = self.declare_parameter("drive_topic", "default").value

        self.base_lookahead = 1.5
        self.base_speed = 2.0
        self.wheelbase_length = 0.33  # Between front and rear axles

        self.trajectory = LineTrajectory("/followed_trajectory")
        self.traj_points = np.array([])  # Empty
        self.is_active = False

        self.traj_sub = self.create_subscription(PoseArray, "/trajectory/current", self.trajectory_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.pose_callback, 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 1)
        self.drive_cmd = AckermannDrive()
        self.drive(0.0, 0.0)

        self.debug_nearest_segment_pub = self.create_publisher(Marker, "/pure_pursuit/nearest_segment", 1)
        self.debug_lookahead_point_pub = self.create_publisher(Marker, "/pure_pursuit/lookahead_point", 1)
        self.debug_lookahead_circle_pub = self.create_publisher(Marker, "/pure_pursuit/lookahead_circle", 1)
        self.debug_text_pub = self.create_publisher(Marker, "/trajectory/debug_text", 1)

        # Track runtime
        self.timing = [0, 0.0]

        self.get_logger().info("Trajectory follower initialized")
        self.run_tests()  # TODO debug

    def get_nearest_segment(self, car_loc: np.ndarray) -> int:
        """Return the segment i s.t. (points[i], points[i+1]) is nearest to the car. car_loc is (x, y)."""
        min_dist = float("inf")
        min_idx = None
        for i in range(len(self.traj_points) - 1):
            dist = point_to_segment_distance(car_loc, self.traj_points[i], self.traj_points[i + 1])
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        return min_idx

    def get_lookahead_point(
        self, car_pose: np.ndarray, lookahead_dist: float, nearest_segment_idx: int
    ) -> Optional[np.ndarray]:
        """Returns the next goal point (x, y), which is a set lookahead dist from the car, or None if no point is found.

        car_pose is (x, y, theta).
        nearest_segment_idx = i is the first segment (points[i], points[i+1]) that's not behind the car.
        Assumes points[i+1] is further along the path than points[i].
        """
        # Find the first segment ahead of the car that intersects the circle
        for i in range(nearest_segment_idx, len(self.traj_points) - 1):
            s1, s2 = self.traj_points[i], self.traj_points[i + 1]
            intersections = circle_segment_intersections(car_pose[:2], lookahead_dist, s1, s2)
            # Use the intersection that's in front of the car
            in_front = []
            for p in intersections:
                car_forward_vec = np.array([math.cos(car_pose[2]), math.sin(car_pose[2])])
                if np.dot(car_forward_vec, p - car_pose[:2]) >= 0:  # Angle between is in [-pi/2, pi/2]
                    in_front.append(p)
            # If both are in front, use the one closer to the end point of the segment
            in_front.sort(key=lambda p: np.linalg.norm(p - s2))
            if in_front:
                return in_front[0]
        # No lookahead point found
        return None

    def pose_callback(self, odometry_msg):
        """Called on every pose update. Updates the drive command."""
        if not self.is_active:
            self.drive(use_last_cmd=True)
            return
        call_time = self.get_clock().now()
        car_pose = pose_to_vec(odometry_msg.pose.pose)
        car_x, car_y, car_theta = car_pose
        car_loc = np.array([car_x, car_y])

        # Check if we're at the goal
        allowed_dist = self.base_lookahead
        if np.linalg.norm(self.traj_points[-1] - car_loc) <= allowed_dist:
            self.get_logger().info(f"Goal reached (dist < {allowed_dist}). Stopping.")
            visualize.plot_debug_text("At goal", self.debug_text_pub, color=(0.0, 0.0, 1.0))
            self.drive(0.0, 0.0)
            self.is_active = False
            return

        # Find the point on the trajectory nearest to the car
        nearest_segment_idx = self.get_nearest_segment(car_loc)

        # Find the lookahead point
        lookahead_point = self.get_lookahead_point(car_pose, self.base_lookahead, nearest_segment_idx)

        if lookahead_point is None:
            # Use last drive command
            self.get_logger().warn("No lookahead point found")
            self.drive(use_last_cmd=True)
            visualize.plot_debug_text("No lookahead point", self.debug_text_pub)
        else:
            visualize.clear_marker(self.debug_text_pub)
            # Drive towards the lookahead point with Ackermann steering
            dx, dy = lookahead_point[0] - car_x, lookahead_point[1] - car_y
            # Converting from global to robot frame (opposite of odometry from robot to global frame)
            local_x = math.cos(-car_theta) * dx - math.sin(-car_theta) * dy
            local_y = math.sin(-car_theta) * dx + math.cos(-car_theta) * dy
            if local_x < 0:
                self.get_logger().warn("Lookahead point behind the vehicle")

            eta = math.atan2(local_y, local_x)
            steering_angle = math.atan((2 * self.wheelbase_length * math.sin(eta)) / self.base_lookahead)
            self.drive(steering_angle, self.base_speed)

        # Debug
        # self.get_logger().info(f"Nearest point: {nearest_point}")
        # self.get_logger().info(f"Lookahead point: {lookahead_point}")
        visualize.plot_line(
            self.traj_points[[nearest_segment_idx, nearest_segment_idx + 1], 0],
            self.traj_points[[nearest_segment_idx, nearest_segment_idx + 1], 1],
            self.debug_nearest_segment_pub,
            color=(0.7, 0.7, 0),
            scale=0.3,
            z=0.025,
            frame="/map",
        )
        if lookahead_point is not None:
            visualize.plot_point(
                lookahead_point[0],
                lookahead_point[1],
                self.debug_lookahead_point_pub,
                color=(0, 1, 0),
                frame="/map",
            )
        else:
            visualize.clear_marker(self.debug_lookahead_point_pub)
        visualize.plot_circle(
            car_x,
            car_y,
            self.base_lookahead,
            self.debug_lookahead_circle_pub,
            z=0.05,
            frame="/map",
        )

        latency = (self.get_clock().now() - call_time).nanoseconds / 1e9
        self.timing[0] += 1
        self.timing[1] += latency
        if self.timing[0] == 50:
            avg_latency = self.timing[1] / 50
            if avg_latency > 0.01:
                self.get_logger().warning(f"high pure pursuit latency, optimize: {avg_latency:.4f}s")
            self.timing = [0, 0.0]

    def trajectory_callback(self, msg):
        """Called when a new trajectory is published."""
        self.get_logger().info(f"Received new trajectory with {len(msg.poses)} points")
        if len(msg.poses) < 2:
            self.get_logger().warn("Trajectory has less than 2 points. Ignoring.")
            return
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        self.traj_points = np.array(self.trajectory.points)  # N x 2
        self.is_active = True

    def drive(self, steering_angle=0.0, velocity=0.0, use_last_cmd=False):
        """
        Publishes to the drive topic with the given steering angle (radians) and velocity (m/s).
        If `use_last_cmd` is set, uses the last drive command.
        """
        header = Header(stamp=self.get_clock().now().to_msg())
        if not use_last_cmd:
            self.drive_cmd = AckermannDrive(
                steering_angle=steering_angle,
                steering_angle_velocity=0.0,
                speed=velocity,
                acceleration=0.0,
                jerk=0.0,
            )
        self.drive_pub.publish(AckermannDriveStamped(header=header, drive=self.drive_cmd))

    def run_tests(self):
        # point_to_segment_distance
        assert np.isclose(point_to_segment_distance(np.array([4, 2]), np.array([2, 1]), np.array([8, 4])), 0)
        assert np.isclose(point_to_segment_distance(np.array([4, 2]), np.array([2, 1]), np.array([4, 2])), 0)
        assert np.isclose(point_to_segment_distance(np.array([4, 2]), np.array([2, 1]), np.array([2, 1])), 2.2360679775)
        assert np.isclose(
            point_to_segment_distance(np.array([4.5, 2.3]), np.array([-0.7, 1.3]), np.array([8.1, 1.3])), 1
        )
        assert np.isclose(
            point_to_segment_distance(np.array([4, 2]), np.array([1, 1]), np.array([3, 1])), 1.41421356237
        )
        assert np.isclose(
            point_to_segment_distance(np.array([-5, 3]), np.array([9, 61]), np.array([-10, -91])), 6.69787566782
        )

        # circle_segment_intersections
        assert np.allclose(
            circle_segment_intersections(np.array([-5, 2]), 3, np.array([0, 4]), np.array([-10, 104])),
            np.empty((0, 2), dtype=float),
        )  # 0 points
        output = circle_segment_intersections(np.array([-5, 2]), 3, np.array([10, 65]), np.array([-10, -15])).flatten()
        output.sort()
        expected = np.array([[-5, 5], [-109 / 17, -11 / 17]]).flatten()
        expected.sort()
        assert np.allclose(output, expected)  # 2 points
        assert np.allclose(
            circle_segment_intersections(np.array([3, 4]), 5, np.array([8, 2]), np.array([8, 5])),
            np.array([[8, 4]]),
        )  # 1 point
        assert np.allclose(
            circle_segment_intersections(np.array([3, 4]), 5, np.array([8, 4]), np.array([8, 4])),
            np.array([[8, 4]]),
        )  # Single point segment
        assert np.allclose(
            circle_segment_intersections(np.array([3.2, 4.6]), 0, np.array([3.5, 4.6]), np.array([-0.7, 4.6])),
            np.array([[3.2, 4.6]]),
        )  # Single point circle
        self.get_logger().info("All tests passed!")


def pose_to_vec(msg: Pose) -> np.ndarray:
    """Convert a Pose message to a 3D vector [x, y, yaw]."""
    quaternion = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    _, _, yaw = tf_transformations.euler_from_quaternion(quaternion)
    return np.array([msg.position.x, msg.position.y, yaw])


def point_to_segment_distance(p: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> float:
    """Returns the minimum distance from a 2D point p to the line segment from s1 to s2."""
    l2 = np.dot(s2 - s1, s2 - s1)  # Squared length of segment
    if np.allclose(l2, 0):
        return np.linalg.norm(p - s1)  # Single point

    t = np.dot(p - s1, s2 - s1) / l2  # Projection of p onto s1s2
    t = np.clip(t, 0, 1)  # So the projection is on the segment

    projection = s1 + t * (s2 - s1)  # Closest point on the segment
    return np.linalg.norm(p - projection)  # Distance to that closest point


def circle_segment_intersections(c: np.ndarray, r: float, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Returns intersection points (0, 1, or 2, (x, y)) of a circle (center c and radius r) and a line segment."""
    # Special case: Segment is a single point
    if np.allclose(s1, s2):
        if np.isclose(np.linalg.norm(s1 - c), r):
            return np.array([s1])  # Point on circle
        else:
            return np.empty((0, 2), dtype=float)  # Point not on circle

    # Solve at^2 + bt + c = 0, where t sweeps along the segment
    # Point = s1 + t * v (0 <= t <= 1)
    v = s2 - s1  # Vector along line segment
    a = np.dot(v, v)
    b = 2 * np.dot(v, s1 - c)
    c = np.dot(s1, s1) + np.dot(c, c) - 2 * np.dot(s1, c) - r**2

    disc = b**2 - 4 * a * c  # Discriminant
    if disc < 0:  # No intersection
        return np.empty((0, 2), dtype=float)

    intersections = []
    # Up to two possible intersections
    sqrt_disc = np.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    if 0 <= t1 <= 1:
        intersections.append(s1 + t1 * v)  # First point
    if 0 <= t2 <= 1 and not np.isclose(t2, t1):
        intersections.append(s1 + t2 * v)  # Second point that's not a duplicate

    if intersections:
        return np.array(intersections)
    else:
        return np.empty((0, 2), dtype=float)


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
