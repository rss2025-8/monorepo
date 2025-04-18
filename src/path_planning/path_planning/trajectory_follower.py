import math
import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import rclpy
import tf_transformations
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.interpolate import splev, splprep
from std_msgs.msg import Float32, Header
from visualization_msgs.msg import Marker

from . import visualize
from .utils import LineTrajectory


class PurePursuit(Node):
    """Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed."""

    def __init__(self):
        super().__init__("trajectory_follower")
        self.odom_topic: str = self.declare_parameter("odom_topic", "default").value
        self.drive_topic: str = self.declare_parameter("drive_topic", "default").value
        self.debug: bool = self.declare_parameter("debug", False).value

        self.max_speed: float = self.declare_parameter("max_speed", 4.0).value
        self.speed: float = self.max_speed
        self.low_pass_filter = LowPassFilter(self.get_clock().now(), cutoff_freq=5.0)

        self.wheelbase_length = 0.33  # Between front and rear axles
        self.max_steering_angle = 0.34
        self.trajectory = LineTrajectory(self, "/followed_trajectory")
        self.traj_points = np.array([])  # Empty
        self.is_active = False

        self.traj_sub = self.create_subscription(PoseArray, "/trajectory/current", self.trajectory_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.pose_callback, 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 1)
        self.drive_cmd = AckermannDrive()

        self.debug_nearest_segment_pub = self.create_publisher(Marker, "/pure_pursuit/nearest_segment", 1)
        self.debug_lookahead_point_pub = self.create_publisher(Marker, "/pure_pursuit/lookahead_point", 1)
        self.debug_lookahead_circle_pub = self.create_publisher(Marker, "/pure_pursuit/lookahead_circle", 1)
        self.debug_driving_arc_pub = self.create_publisher(Marker, "/pure_pursuit/driving_arc", 1)
        self.debug_drive_line_pub = self.create_publisher(Marker, "/pure_pursuit/drive_line", 1)
        self.debug_text_pub = self.create_publisher(Marker, "/trajectory/debug_text", 1)

        # Track runtime
        self.timing = [0, 0.0]

        if self.debug:
            self.get_logger().info("DEBUG mode enabled")
        self.get_logger().info("Trajectory follower initialized")
        # if self.debug:
        #     self.run_tests()

        self.pose_to_traj_error_pub = self.create_publisher(Float32, "/pose_to_traj_error", 1)

    def get_nearest_segment(self, car_loc: np.ndarray) -> int:
        """Return the segment i s.t. (points[i], points[i+1]) is nearest to the car. car_loc is (x, y)."""
        car_locs = np.tile(car_loc, (len(self.traj_points) - 1, 1))  # N x 2, N (# of segments) copies of car_loc
        dists = vectorized_point_to_segment_distance(car_locs, self.traj_points[:-1], self.traj_points[1:])
        return np.argmin(dists)

    def get_lookahead_point(
        self, car_pose: np.ndarray, lookahead_dist: float, nearest_segment_idx: int
    ) -> Optional[Tuple[np.ndarray, int]]:
        """Returns the next goal point (x, y), which is a set lookahead dist from the car, or None if no point is found.
        Also returns the index of the segment the lookahead point is on.

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
                if np.dot(car_forward_vec, p - car_pose[:2]) >= -1e-6:  # Angle between is in [-pi/2, pi/2]
                    in_front.append(p)
            # If both are in front, use the one closer to the end point of the segment
            in_front.sort(key=lambda p: np.linalg.norm(p - s2))
            if in_front:
                return in_front[0], i
        # No lookahead point found
        return None

    def get_adaptive_lookahead(self, car_pose: np.ndarray, nearest_segment_idx: int) -> float:
        """Returns the adaptive lookahead distance for the given car pose and nearest segment index."""
        # Tuned lookahead distances
        max_lookahead = 3.0 + (self.max_speed - 1) * 0.75
        # max_lookahead = 3.0 + (self.max_speed - 1) * 1.0  # Drifts a bit too far from the path at high speeds
        # min_lookahead = 0.75 + (self.max_speed - 1) * 0.25  # Distance near the car to ignore, a bit too low
        min_lookahead = 0.75 + (self.max_speed - 1) * 0.3  # Distance near the car to ignore
        # min_lookahead = 0.75 + (self.max_speed - 1) * 0.35  # Distance near the car to ignore, too high
        dtheta_threshold = 0.75  # Curvature of turn needed to stop expanding lookahead

        # Lookahead as large as possible until a sharp turn is detected
        # Find sum of change in dtheta between current and future segments
        dtheta_sum = 0.0
        curr_dist = np.linalg.norm(self.traj_points[nearest_segment_idx + 1] - car_pose[:2])
        # Sum angles a bit before current segment to keep lookahead shorter coming out of a turn
        meters_before = self.max_speed * 0.25
        # meters_before = 0
        num_before = min(int(meters_before / self.traj_step_size), nearest_segment_idx)
        curr_dist -= self.traj_step_size * num_before
        # Determine lookahead distance based on curvature
        cutoff_idx = len(self.traj_dtheta) - 1
        for i in range(nearest_segment_idx + 1 - num_before, len(self.traj_dtheta)):
            curr_dist += self.traj_step_size
            dtheta_sum += self.traj_dtheta[i]
            if abs(dtheta_sum) > dtheta_threshold or curr_dist - self.traj_step_size > max_lookahead:
                cutoff_idx = i
                break
        # Set lookahead to be the distance to this segment
        adaptive_lookahead = np.clip(
            np.linalg.norm(self.traj_points[cutoff_idx] - car_pose[:2]), min_lookahead, max_lookahead
        )
        return adaptive_lookahead

    def pose_callback(self, odometry_msg):
        """Called on every pose update. Updates the drive command."""
        if not self.is_active:
            self.drive(use_last_cmd=True)
            return
        call_time = self.get_clock().now()
        car_pose = pose_to_vec(odometry_msg.pose.pose)
        car_x, car_y, car_theta = car_pose
        car_loc = np.array([car_x, car_y])

        # Find the point on the trajectory nearest to the car
        nearest_segment_idx = self.get_nearest_segment(car_loc)

        # Check if we're at the goal (2nd to last point or closest segment is the one after the goal)
        allowed_dist = self.speed * 0.5
        if (
            nearest_segment_idx == len(self.traj_points) - 2
            or np.linalg.norm(self.traj_points[-2] - car_loc) <= allowed_dist
        ):
            if nearest_segment_idx == len(self.traj_points) - 2:
                self.get_logger().info(f"Goal reached (nearest to end segment). Stopping.")
            else:
                self.get_logger().info(f"Goal reached (dist < {allowed_dist}). Stopping.")
            if self.debug:
                visualize.plot_debug_text("At goal", self.debug_text_pub, color=(0.0, 0.0, 1.0))
            self.drive(0.0, 0.0)
            self.is_active = False
            return

        # Find adaptive lookahead distance using curvature
        adaptive_lookahead = self.get_adaptive_lookahead(car_pose, nearest_segment_idx)

        # Find the lookahead point
        result = self.get_lookahead_point(car_pose, adaptive_lookahead, nearest_segment_idx)
        lookahead_point, l_seg_idx = result if result else (None, None)

        if lookahead_point is None:
            # Try to get back on the path
            # self.get_logger().warning("No lookahead point found, using end point of closest segment")
            l_seg_idx = nearest_segment_idx
            lookahead_point = self.traj_points[l_seg_idx + 1]
            self.drive(use_last_cmd=True)
            if self.debug:
                visualize.plot_debug_text("No lookahead point", self.debug_text_pub)
        elif self.debug:
            visualize.clear_marker(self.debug_text_pub)

        # Drive towards the lookahead point with Ackermann steering
        dx, dy = lookahead_point[0] - car_x, lookahead_point[1] - car_y
        # Converting from global to robot frame (opposite of odometry from robot to global frame)
        local_x = math.cos(-car_theta) * dx - math.sin(-car_theta) * dy
        local_y = math.sin(-car_theta) * dx + math.cos(-car_theta) * dy

        eta = math.atan2(local_y, local_x)
        if local_x < 0:
            # Commit to a hard turn
            if eta == 0:
                eta = 1e-6
            eta = math.pi / 2 * np.sign(eta)
        steering_angle = math.atan((2 * self.wheelbase_length * math.sin(eta)) / adaptive_lookahead)

        # Set speed based on allowed velocity to lookahead distance ratio
        min_speed = 1.0
        min_v_to_lookahead_ratio = 0.75
        self.speed = np.clip(adaptive_lookahead / min_v_to_lookahead_ratio, min_speed, self.max_speed)

        # Low pass filter to smooth out sudden unexpected spikes
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        steering_angle = self.low_pass_filter.update(steering_angle, self.get_clock().now())
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)

        # Draw circle that the car is following
        if self.debug and eta != 0:
            R = adaptive_lookahead / (2 * math.sin(eta))
            visualize.plot_circle(0, R, R, self.debug_driving_arc_pub, color=(0, 0, 1), frame="/base_link")

        self.drive(steering_angle, self.speed)
        self.publish_pose_to_traj_error(car_loc, nearest_segment_idx)

        # Debug
        # self.get_logger().info(f"Nearest point: {nearest_point}")
        # self.get_logger().info(f"Lookahead point: {lookahead_point}")
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
        if self.debug:
            visualize.plot_line(
                self.traj_points[[nearest_segment_idx, nearest_segment_idx + 1], 0],
                self.traj_points[[nearest_segment_idx, nearest_segment_idx + 1], 1],
                self.debug_nearest_segment_pub,
                color=(0.7, 0.7, 0),
                scale=0.3,
                z=0.025,
                frame="/map",
            )
            visualize.plot_circle(
                car_x,
                car_y,
                adaptive_lookahead,
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
            else:
                self.get_logger().info(f"pp: {avg_latency:.4f}s")
            self.timing = [0, 0.0]

    def smooth_path(self, path: np.ndarray, num_points: int, smoothness: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Smooths a path using a spline. Returns the path, an estimate of dtheta, and the step size.

        Args:
            path: The path to smooth.
            num_points: The number of points in the smoothed path.
            smoothness: The smoothness of the spline.
        """
        # First, split the path into linear segments
        path = split_up_path(path, num_points)

        # Parameterize by distance along the path
        dx = np.diff(path[:, 0])  # N-1
        dy = np.diff(path[:, 1])  # N-1
        step_size = np.hypot(dx[-1], dy[-1])
        dist = np.hstack([0, np.cumsum(np.hypot(dx, dy))])  # N
        u = dist / dist[-1]  # Normalize distances to [0, 1]

        # Fit a B-spline to the path
        tck, _ = splprep([path[:, 0], path[:, 1]], u=u, s=smoothness)

        # Sample uniformly spaced points along the path
        u_samples = np.linspace(0, 1, num_points)
        x_samples, y_samples = splev(u_samples, tck)

        # Find curvature at each point
        dx, dy = splev(u_samples, tck, der=1)
        ddx, ddy = splev(u_samples, tck, der=2)
        kappa = abs((dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5)  # (1 / radius of curve)
        dtheta = kappa * step_size

        return np.vstack([x_samples, y_samples]).T, dtheta, step_size

    def trajectory_callback(self, msg):
        """Called when a new trajectory is published."""
        self.get_logger().info(f"Received new trajectory with {len(msg.poses)} points")
        if len(msg.poses) < 2:
            self.get_logger().warning("Trajectory has less than 2 points. Ignoring.")
            return
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)

        # Generate a smoothed trajectory
        dist_between_points = 0.25
        path = np.array(self.trajectory.points)
        path_length = np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1))
        num_points = int(path_length / dist_between_points) + 2
        path, dtheta, step_size = self.smooth_path(path, num_points=num_points, smoothness=0.25)

        self.trajectory.clear()
        for point in path:
            self.trajectory.addPoint(point)
        self.trajectory.publish_viz()
        self.traj_points = path
        self.traj_dtheta = dtheta
        self.traj_step_size = step_size
        # Add an extra point at the end of the trajectory that extends the last segment
        norm_vec = self.traj_points[-1] - self.traj_points[-2]
        norm_vec = norm_vec / np.linalg.norm(norm_vec)
        end_point = self.traj_points[-1] + norm_vec * self.max_speed * 10
        self.traj_points = np.concatenate([self.traj_points, [end_point]])
        self.traj_dtheta = np.concatenate([self.traj_dtheta, [0.0]])
        self.is_active = True
        self.get_logger().info(f"Final trajectory has {len(self.traj_points)} points")

    def drive(self, steering_angle=0.0, velocity=0.0, use_last_cmd=False):
        """
        Publishes to the drive topic with the given steering angle (radians) and velocity (m/s).
        If `use_last_cmd` is set, uses the last drive command.
        """
        if abs(steering_angle) > math.pi / 2:
            self.get_logger().warning(f"Steering angle {steering_angle} is too large")
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)

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

        if self.debug:
            visualize.plot_line(
                [0.0, math.cos(steering_angle / self.max_steering_angle * math.pi / 2) * velocity / self.max_speed],
                [0.0, math.sin(steering_angle / self.max_steering_angle * math.pi / 2) * velocity / self.max_speed],
                self.debug_drive_line_pub,
                color=(0.0, 0.0, 1.0),
                z=0.1,
            )

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

        # point_to_segments_distances
        # Combine all tests above
        assert np.allclose(
            vectorized_point_to_segment_distance(
                np.array([[4, 2], [4, 2], [4, 2], [4.5, 2.3], [4, 2], [-5, 3]]),
                np.array([[2, 1], [2, 1], [2, 1], [-0.7, 1.3], [1, 1], [9, 61]]),
                np.array([[8, 4], [4, 2], [2, 1], [8.1, 1.3], [3, 1], [-10, -91]]),
            ),
            np.array([0, 0, 2.2360679775, 1, 1.41421356237, 6.69787566782]),
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

        # fraction_along_segment
        assert np.isclose(fraction_along_segment(np.array([0, 0]), np.array([1, 0]), np.array([0.7, 0])), 0.7)
        assert np.isclose(fraction_along_segment(np.array([-2, -4]), np.array([3, 6]), np.array([-2, -4])), 0.0)
        assert np.isclose(fraction_along_segment(np.array([-2, -4]), np.array([3, 6]), np.array([3, 6])), 1.0)
        assert np.isclose(fraction_along_segment(np.array([-2, -4]), np.array([3, 6]), np.array([-1, -2])), 0.2)

        self.get_logger().info("All tests passed!")

    def publish_pose_to_traj_error(self, car_pose, nearest_segment_idx):
        s1 = self.traj_points[nearest_segment_idx]
        s2 = self.traj_points[nearest_segment_idx + 1]

        error = point_to_segment_distance(car_pose, s1, s2)

        error_msg = Float32()
        error_msg.data = float(error)
        self.pose_to_traj_error_pub.publish(error_msg)


class LowPassFilter:
    """A first-order exponential low-pass filter."""

    def __init__(self, time: rclpy.time.Time, cutoff_freq: float):
        """
        Creates the filter with the given initial time and cutoff frequency (Hz).
        """
        self.prev_time = time
        self.fc = cutoff_freq
        self.prev_output = 0.0

    def update(self, input_val: float, time: rclpy.time.Time) -> float:
        """
        Update filter state with the current input value and its timestamp.
        Returns the filtered output.
        """
        dt = (time.nanoseconds - self.prev_time.nanoseconds) / 1e9
        alpha = math.exp(-2 * math.pi * self.fc * dt)
        y = alpha * self.prev_output + (1 - alpha) * input_val
        self.prev_output = y
        self.prev_time = time
        return y


def pose_to_vec(msg: Pose) -> np.ndarray:
    """Convert a Pose message to a 3D vector [x, y, yaw]."""
    quaternion = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    _, _, yaw = tf_transformations.euler_from_quaternion(quaternion)
    return np.array([msg.position.x, msg.position.y, yaw])


def point_to_segment_distance(p: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> float:
    """Returns the minimum distance from a 2D point p to the line segment from s1 to s2."""
    if np.linalg.norm(s1 - s2) < 1e-6:
        return np.linalg.norm(p - s1)  # Single point

    l2 = np.dot(s2 - s1, s2 - s1)  # Squared length of segment
    t = np.dot(p - s1, s2 - s1) / l2  # Projection of p onto s1s2
    t = np.clip(t, 0, 1)  # So the projection is on the segment

    projection = s1 + t * (s2 - s1)  # Closest point on the segment
    return np.linalg.norm(p - projection)  # Distance to that closest point


def vectorized_point_to_segment_distance(P: np.ndarray, S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """Returns the minimum distance from point P[i] to the segment (S1[i], S2[i]) for all i.

    P should be N x 2, S1 and S2 should be N x 2. Returns a 1D array of length N.
    """
    assert P.shape[0] == S1.shape[0] == S2.shape[0]
    assert P.shape[1] == 2 and S1.shape[1] == 2 and S2.shape[1] == 2

    diff = S2 - S1  # N x 2
    L2 = np.sum(diff * diff, axis=1)  # N
    L2 = np.where(L2 > 0, L2, 1.0)  # N (Avoid division by zero)

    # Projection of p onto s1s2 for each segment
    t = np.sum((P - S1) * diff, axis=1) / L2
    t = np.clip(t, 0.0, 1.0)  # N
    # Closest points on segments
    projection = S1 + diff * t[:, None]  # N x 2
    # Distances to closest points
    return np.linalg.norm(P - projection, axis=1)


def circle_segment_intersections(c: np.ndarray, r: float, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Returns intersection points (0, 1, or 2, (x, y)) of a circle (center c and radius r) and a line segment."""
    # Special case: Segment is a single point
    if np.linalg.norm(s1 - s2) < 1e-6:
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
    if disc < -1e-6:  # No intersection
        return np.empty((0, 2), dtype=float)
    elif disc < 0:
        disc = 0  # Numerical tolerance

    intersections = []
    # Up to two possible intersections
    sqrt_disc = np.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    if -1e-6 <= t1 <= 1 + 1e-6:
        intersections.append(s1 + t1 * v)  # First point
    if -1e-6 <= t2 <= 1 + 1e-6 and abs(t2 - t1) > 1e-6:
        intersections.append(s1 + t2 * v)  # Second point that's not a duplicate

    if intersections:
        return np.array(intersections)
    else:
        return np.empty((0, 2), dtype=float)


def fraction_along_segment(s1: np.ndarray, s2: np.ndarray, p: np.ndarray) -> float:
    """
    Return how far along p is on the segment (s1, s2), as a number in [0, 1].
    Assumes p lies on the line segment.
    """
    v = s2 - s1
    t = np.dot(p - s1, v) / np.dot(v, v)  # Normalized projection of p onto s1s2
    return float(t)


def split_up_path(path: np.ndarray, num_points: int) -> np.ndarray:
    """Splits a path into num_points points, uniformly spaced along the path.

    Args:
        path: The path to split (N x 2).
        num_points: The number of points to split the path into.
    Returns:
        The split path (M x 2).
    """
    dx = np.diff(path[:, 0])  # N-1
    dy = np.diff(path[:, 1])  # N-1
    dist = np.hstack([0, np.cumsum(np.hypot(dx, dy))])  # N
    u = np.linspace(0, dist[-1], num_points)
    return np.vstack([np.interp(u, dist, path[:, 0]), np.interp(u, dist, path[:, 1])]).T


# # For profiling
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
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
