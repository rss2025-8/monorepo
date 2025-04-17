"""
Uses PRM.
Tunable parameters: kernel_size (dilation of obstacles), N (number of samples)
"""

import math

import cv2
import rclpy
from rclpy.node import Node

assert rclpy
import queue

import numpy as np
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from scipy.interpolate import splev, splprep
from scipy.spatial import KDTree
from visualization_msgs.msg import Marker

from . import visualize
from .utils import LineTrajectory


class PathPlan(Node):
    """Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.odom_topic: str = self.declare_parameter("odom_topic", "default").value
        self.map_topic: str = self.declare_parameter("map_topic", "default").value
        self.initial_pose_topic: str = self.declare_parameter("initial_pose_topic", "default").value
        self.debug: bool = self.declare_parameter("debug", False).value

        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_cb, 1)
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 10)
        self.traj_pub = self.create_publisher(PoseArray, "/trajectory/current", 10)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, self.initial_pose_topic, self.pose_cb, 10)
        self.neighbors_pub = self.create_publisher(PoseArray, "/trajectory/end_point_neighbors", 10)

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory" if self.debug else None)
        self.grid = None
        self.dist_to_obstacle_grid = None
        self.downsample_factor = 4
        self.resolution = None
        self.origin_x = None
        self.origin_y = None

        self.pose_x = None
        self.pose_y = None
        self.goal_x = None
        self.goal_y = None

        self.debug_text_pub = self.create_publisher(Marker, "/trajectory/debug_text", 1)
        visualize.clear_marker(self.debug_text_pub)

        self.get_logger().info("Trajectory planner initialized")

    def map_cb(self, msg):
        # Reshape the OccupancyGrid
        self.grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        obstacle_mask = (self.grid != 0).astype(np.uint8)  # 0 is free, -1 is unknown, 100 is obstacle

        # Dilate the obstacles with a kernel (size controls buffer)
        kernel_size = 10  # TODO play with this value
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_obstacle_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)

        # Update grid: obstacles=100, free = original value
        self.grid = np.where(dilated_obstacle_mask == 1, 100, self.grid)
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

        # Precompute distance to the nearest obstacle for each cell with BFS, in meters
        # Downsample for faster computation
        self.get_logger().info(f"Precomputing distance to obstacle grid (downsampled by {self.downsample_factor})...")
        small_grid = cv2.resize(
            self.grid,
            (self.grid.shape[1] // self.downsample_factor, self.grid.shape[0] // self.downsample_factor),
            interpolation=cv2.INTER_NEAREST,
        )
        self.dist_to_obstacle_grid = np.full_like(small_grid, np.inf, dtype=np.float32)
        visited = np.zeros_like(small_grid, dtype=bool)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # Starting states
        q = queue.Queue()
        for i in range(small_grid.shape[0]):
            for j in range(small_grid.shape[1]):
                if small_grid[i, j] == 100:
                    q.put((i, j))
                    visited[i, j] = True
                    self.dist_to_obstacle_grid[i, j] = 0
        # BFS
        while not q.empty():
            i, j = q.get()
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if (
                    0 <= ni < small_grid.shape[0]
                    and 0 <= nj < small_grid.shape[1]
                    and small_grid[ni, nj] != 100
                    and not visited[ni, nj]
                ):
                    visited[ni, nj] = True
                    self.dist_to_obstacle_grid[ni, nj] = (
                        self.dist_to_obstacle_grid[i, j] + self.resolution * self.downsample_factor
                    )
                    q.put((ni, nj))

        self.get_logger().info(f"Processed map data of shape {self.grid.shape}, dilation {kernel_size}")
        if self.debug:
            visualize.plot_debug_text("Ready", self.debug_text_pub, color=(0.0, 0.0, 1.0))

    def pose_cb(self, pose):
        self.pose_x = pose.pose.pose.position.x
        self.pose_y = pose.pose.pose.position.y
        # self.pose_x = 10.356518745422363
        # self.pose_y = -1.18073570728302
        self.get_logger().info(f"Received current pose: {self.pose_x}, {self.pose_y}")
        self.plan_path((self.pose_x, self.pose_y), (self.goal_x, self.goal_y), self.grid)

    def goal_cb(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        # self.goal_x = -18.909656524658203
        # self.goal_y = 7.085318565368652
        self.get_logger().info(f"Received goal pose: {self.goal_x}, {self.goal_y}")

    def plan_path(self, start_point, end_point, map):
        """Plan a path from start to end point using PRM (sampling-based method)."""
        self.trajectory.clear()
        if self.grid is None:
            self.get_logger().warning(f"NO MAP! (Relaunch the map)")
            if self.debug:
                visualize.plot_debug_text("No map (relaunch)", self.debug_text_pub)
            return

        if start_point[0] is None:
            self.get_logger().warning(f"NO START POINT!")
            if self.debug:
                visualize.plot_debug_text("No start point", self.debug_text_pub)
            return

        if end_point[0] is None:
            self.get_logger().warning(f"NO GOAL POINT!")
            if self.debug:
                visualize.plot_debug_text("No goal point", self.debug_text_pub)
            return

        if self.debug:
            visualize.clear_marker(self.debug_text_pub)

        def is_free(x, y):
            """Check if a point (x, y) is free in the grid."""
            grid_x = min(int((self.origin_x - x) / self.resolution), self.grid.shape[1] - 1)
            grid_y = min(int((self.origin_y - y) / self.resolution), self.grid.shape[0] - 1)

            if 0 <= grid_x < self.grid.shape[1] and 0 <= grid_y < self.grid.shape[0]:
                return self.grid[grid_y, grid_x] == 0
            return False

        def sample_free():
            """Sample a random free point in the grid."""
            # TODO can be faster by sampling all points at once, checking if they're free vectorized
            while True:
                x_max = self.origin_x
                x_min = self.origin_x - self.grid.shape[1] * self.resolution
                y_max = self.origin_y
                y_min = self.origin_y - self.grid.shape[0] * self.resolution

                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)

                if is_free(x, y):
                    return (x, y)

        def is_collision_free(p1, p2):
            """Check if a line segment between two points is collision free."""
            # TODO can optimize with range_libc
            x1, y1 = p1
            x2, y2 = p2
            dx = x2 - x1
            dy = y2 - y1
            dist = math.hypot(dx, dy)
            steps = int(dist / 0.5) + 1
            for i in range(steps + 1):
                t = i / steps
                x = x1 + t * dx
                y = y1 + t * dy
                if not is_free(x, y):
                    return False
            return True

        def segment_dist_to_obstacle(p1, p2):
            """Check approximately how close a line segment gets to an obstacle."""
            # TODO can optimize with range_libc
            x1, y1 = p1
            x2, y2 = p2
            dx = x2 - x1
            dy = y2 - y1
            dist = math.hypot(dx, dy)
            steps = int(dist / 0.5 * 4) + 1  # Approximate is fine
            min_dist = np.inf
            for i in range(steps + 1):
                t = i / steps
                x = x1 + t * dx
                y = y1 + t * dy
                grid_x = min(
                    int((self.origin_x - x) / self.resolution / self.downsample_factor),
                    self.dist_to_obstacle_grid.shape[1] - 1,
                )
                grid_y = min(
                    int((self.origin_y - y) / self.resolution / self.downsample_factor),
                    self.dist_to_obstacle_grid.shape[0] - 1,
                )
                min_dist = min(min_dist, self.dist_to_obstacle_grid[grid_y, grid_x])
            return min_dist

        N = 1000  # TODO number of samples
        start_time = self.get_clock().now()
        points = [sample_free() for _ in range(N)]
        points.append(start_point)
        points.append(end_point)
        self.get_logger().info(f"Sampling {N} points: {(self.get_clock().now() - start_time).nanoseconds / 1e9:.3f}s")

        # Build map
        graph = {point: [] for point in points}
        kdtree = KDTree(points)

        # Sample points and build the graph
        start_time = self.get_clock().now()
        for i, point in enumerate(points):
            distances, indices = kdtree.query(point, k=40)
            # for the start point, print out the coordinates of the neighbors
            for j in indices:
                if point == points[j]:
                    continue
                neighbor = points[j]
                if is_collision_free(point, neighbor):
                    graph[point].append(neighbor)
        # self.get_logger().info("Graph: " + str(graph[end_point]))
        neighbors_pose_array = PoseArray()
        neighbors_pose_array.header.frame_id = "map"  # Set the appropriate frame ID
        neighbors_pose_array.header.stamp = self.get_clock().now().to_msg()
        for neighbor in points:
            pose = Pose()
            pose.position.x = neighbor[0]
            pose.position.y = neighbor[1]
            pose.position.z = 0.0
            neighbors_pose_array.poses.append(pose)
        if self.debug and self.neighbors_pub.get_subscription_count() > 0:
            self.neighbors_pub.publish(neighbors_pose_array)
        self.get_logger().info(f"Building graph: {(self.get_clock().now() - start_time).nanoseconds / 1e9:.3f}s")

        # Build path by implementing A*
        start_time = self.get_clock().now()
        queue = [(0, start_point)]
        costs = {start_point: 0}
        parents = {start_point: None}
        visited = set()
        while queue:
            # self.get_logger().info("Queue: " + str(queue))
            queue.sort(key=lambda x: x[0])
            current_cost, current_point = queue.pop(0)
            if current_point in visited:
                continue
            visited.add(current_point)

            if current_point == end_point:
                # self.get_logger().info("Reached the goal!")
                break

            for neighbor in graph[current_point]:
                # Add a tunable cost based on approximate distance to the nearest obstacle
                # Force is proportional to 1/r
                dist_to_obstacle = segment_dist_to_obstacle(current_point, neighbor)
                # TODO play with these
                potential_field_weight = 1.0
                potential_field_base = 0.25
                new_cost = (
                    costs[current_point]
                    + math.hypot(neighbor[0] - current_point[0], neighbor[1] - current_point[1])
                    + (potential_field_weight / (dist_to_obstacle + potential_field_base))
                )
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    priority = new_cost + math.hypot(end_point[0] - neighbor[0], end_point[1] - neighbor[1])
                    queue.append((priority, neighbor))
                    parents[neighbor] = current_point
        self.get_logger().info(f"A* pathfinding: {(self.get_clock().now() - start_time).nanoseconds / 1e9:.3f}s")
        # self.get_logger().info("Parents: " + str(parents[end_point]))

        # Reconstruct path
        path = []
        current = end_point
        while current is not None:
            path.append(current)
            current = parents.get(current, None)
        path.reverse()
        self.get_logger().info(f"Planned path: {path}")
        for point in path:
            self.trajectory.addPoint(point)

        # Calculate min distance to wall
        minimum_dist = self.find_closest_point_to_wall(path)
        self.get_logger().info(f"Minimum distance to wall: {minimum_dist}")
        if self.debug:
            self.trajectory.publish_viz()
        self.traj_pub.publish(self.trajectory.toPoseArray())

    def find_closest_point_to_wall(self, points):
        """Find distance to the closest point on the wall of the map to the given list of points."""
        min_distance = np.inf
        for point in points:
            x = point[0]
            y = point[1]
            grid_x = min(int(((self.origin_x - x) / self.resolution) / self.downsample_factor), self.grid.shape[1] - 1)
            grid_y = min(int(((self.origin_y - y) / self.resolution) / self.downsample_factor), self.grid.shape[0] - 1)
            if 0 <= grid_x < self.grid.shape[1] and 0 <= grid_y < self.grid.shape[0]:
                distance = self.dist_to_obstacle_grid[grid_y, grid_x]
                if distance < min_distance:
                    min_distance = distance
        return min_distance

    def smooth_path(self, path: np.ndarray, num_points: int, smoothness: float) -> np.ndarray:
        """Smooths a path using a spline.

        Args:
            path: The path to smooth.
            num_points: The number of points in the smoothed path.
            smoothness: The smoothness of the spline.
        """
        # Parameterize by distance along the path
        dx = np.diff(path[:, 0])  # N-1
        dy = np.diff(path[:, 1])  # N-1
        dist = np.hstack([0, np.cumsum(np.hypot(dx, dy))])  # N
        u = dist / dist[-1]  # Normalize distances to [0, 1]

        # Fit a B-spline to the path
        tck, _ = splprep([path[:, 0], path[:, 1]], u=u, s=smoothness)

        # Sample uniformly spaced points along the path
        u_samples = np.linspace(0, 1, num_points)
        x_samples, y_samples = splev(u_samples, tck)
        return np.vstack([x_samples, y_samples]).T  # M x 2


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
