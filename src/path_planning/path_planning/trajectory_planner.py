"""
Uses A* on a 2D grid.
Tunable parameters: Downsampling factor, gaussian blur sigma, added weight for cells near obstacles
"""

import math
import queue
from dataclasses import dataclass
from typing import Iterable

# import dubins
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.ndimage import gaussian_filter

assert rclpy
import geometry_msgs
from geometry_msgs.msg import (
    Pose,
    PoseArray,
    PoseStamped,
    PoseWithCovarianceStamped,
    Quaternion
)
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Header

from .utils import LineTrajectory, load_map


def convolution_downsample(mat: np.ndarray, kernel_size: int):
    """idk what a "numpy" is
    fortunately this is only ran once"""

    height, width = mat.shape
    out_height = (height + kernel_size - 1) // kernel_size
    out_width = (width + kernel_size - 1) // kernel_size

    result = np.zeros((out_height, out_width), dtype=int)

    for i in range(out_height):
        for j in range(out_width):
            # Define the n x n window
            start_i = i * kernel_size
            start_j = j * kernel_size
            end_i = min(start_i + kernel_size, height)
            end_j = min(start_j + kernel_size, width)

            result[i, j] = np.max(mat[start_i:end_i, start_j:end_j])

    return result


@dataclass(frozen=True, order=True)
class Point:
    x: int  # u
    y: int  # v

    @classmethod
    def from_msg(cls, msg: geometry_msgs.msg.Point, grid: "Grid") -> "Point":
        return grid.real_to_grid((msg.x, msg.y))

    def dist(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def as_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)


@dataclass(frozen=True)
class Grid:
    grid: np.ndarray
    resolution: float  # px to m
    origin: Point

    @classmethod
    def empty(cls) -> "Grid":
        return cls(np.empty((1, 1)), 1, Point(0, 0))

    @classmethod
    def from_msg(cls, msg: OccupancyGrid, downsampling: int = 1) -> "Grid":
        # Reshape the OccupancyGrid
        grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        # treat empty space as walls
        grid[grid == -1] = 100
        grid[grid == 100] = 100

        # idk numpy so 2d convolution
        downsampled_grid = convolution_downsample(grid, downsampling)

        # Dilate the obstacles with a kernel (size controls buffer)
        obstacle_mask = (downsampled_grid != 0).astype(np.uint8)  # 0 is free, -1 is unknown, 100 is obstacle
        kernel_size = 8
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_obstacle_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)
        final_grid = np.where(dilated_obstacle_mask == 1, 100, downsampled_grid)
        
        inner_blur = gaussian_filter(downsampled_grid, sigma=3)
        outer_blur = gaussian_filter(downsampled_grid, sigma=9)
        # smooth =

        return cls(
            grid=np.maximum(np.maximum(inner_blur, outer_blur), downsampled_grid),
            resolution=msg.info.resolution * downsampling,
            origin=Point(msg.info.origin.position.x, msg.info.origin.position.y),
        )

    def get_weight(self, point: Point) -> float:
        grid_y = int(point.y)
        grid_x = int(point.x)

        if self.in_bounds(grid_x, grid_y):
            return self.grid[grid_y, grid_x]

        return float("+inf")

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0]

    def get_neighbors(self, point: Point, dist: int = 1) -> Iterable[Point]:
        return (
            Point(point.x - dist, point.y),
            Point(point.x, point.y - dist),
            Point(point.x + dist, point.y),
            Point(point.x, point.y + dist),
        )

    def real_to_grid(self, p: tuple[float, float]) -> Point:
        return Point(int((-p[0] + self.origin.x) / self.resolution), int((-p[1] + self.origin.y) / self.resolution))

    def grid_to_real(self, p: Point) -> tuple[float, float]:
        return (float(self.origin.x - p.x * self.resolution), float(self.origin.y - p.y * self.resolution))

    def get_free_arr(self, resolution: float) -> PoseArray:
        free_coords = np.argwhere(self.grid == 0)
        # Pick out 1000 random points (for visualization)
        if len(free_coords) > 1000:
            free_coords = free_coords[np.random.choice(len(free_coords), 1000, replace=False)]
        return PoseArray(
            header=Header(frame_id="map"),
            poses=[
                Pose(
                    position=geometry_msgs.msg.Point(
                        y=-x * resolution + self.origin.y, x=self.origin.x - y * resolution, z=0.0
                    )
                )
                for x, y in free_coords
            ],
        )


def dubins_dist(start: Point, end: Point) -> float:
    pass


class PathPlan(Node):
    """Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.odom_topic: str = self.declare_parameter("odom_topic", "default").value
        self.map_topic: str = self.declare_parameter("map_topic", "default").value
        self.initial_pose_topic: str = "/pf/pose/odom" # self.declare_parameter("initialposetopic", "default").value
        self.debug: bool = self.declare_parameter("debug", False).value

        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_cb, 1)
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 10)
        self.traj_pub = self.create_publisher(PoseArray, "/trajectory/current", 10)
        self.pose_sub = self.create_subscription(Odometry, self.initial_pose_topic, self.pose_cb, 10)
        self.debug_pub = self.create_publisher(PoseArray, "/debug_map", 10)

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")
        self.last_pose: Pose
        self.last_goal: Pose
        self.grid: Grid = Grid.empty()

        # Load the provided map (if given)
        self.map_set = False
        map_to_load = self.declare_parameter("map_to_load", "").value
        if map_to_load:
            self.get_logger().info(f"Loading map from {map_to_load}")
            self.map_cb(load_map(map_to_load))
        if self.debug:
            self.get_logger().info("DEBUG mode enabled")
        self.get_logger().info(f"A* trajectory planner initialized")

    def map_cb(self, msg: OccupancyGrid):
        if self.map_set:
            self.get_logger().info("Ignoring duplicate map message")
            return
        self.map_set = True
        self.grid = Grid.from_msg(msg, downsampling=3)
        # self.get_logger().info(f"orientation {msg.info.origin.orientation}")
        # self.get_logger().info(
        # f"Received grid: {self.grid}; shape: {self.grid.grid.shape}; neighbors {list(self.grid.get_weight(v) for v in self.grid.get_neighbors(Point(0, 0)))}"
        # )
        self.get_logger().info(f"Received map")
        if self.debug:
            self.debug_pub.publish(self.grid.get_free_arr(self.grid.resolution))

    def pose_cb(self, pose: Odometry):
        self.last_pose = Point.from_msg(pose.pose.pose.position, self.grid)
        # self.get_logger().info(
        #     f"Received pose: {self.last_pose}; ({pose.pose.pose.position.x}, {pose.pose.pose.position.y})"
        # )
        # self.plan_path(self.last_pose, self.last_goal, self.grid)

    def goal_cb(self, msg: PoseStamped):
        self.last_goal = Point.from_msg(msg.pose.position, self.grid)
        # self.last_goal = self.grid.real_to_grid((-18.909656524658203, 7.085318565368652))
        self.get_logger().info(f"Received goal: {self.last_goal}; ({msg.pose.position.x}, {msg.pose.position.y})")
        self.plan_path(self.last_pose, self.last_goal, self.grid)

    def plan_path(self, start_point: Point, end_point: Point, map: Grid):
        start_time = self.get_clock().now()
        self.trajectory.clear()

        agenda = queue.PriorityQueue()
        agenda.put((0.0, start_point))
        traversed = {start_point: None}
        cost_map = {start_point: 0.0}

        i = 0
        while not agenda.empty():
            current: Point = agenda.get()[1]

            if current == end_point:
                break

            for next in map.get_neighbors(current):
                i = i + 1
                weight = map.get_weight(next)
                if weight > 95:
                    continue
                new_cost = cost_map[current] + 1 + weight

                if next not in cost_map or new_cost < cost_map[next]:
                    cost_map[next] = new_cost

                    priority = new_cost + next.dist(end_point)
                    agenda.put((priority, next))
                    traversed[next] = current

        self.get_logger().info(f"Num points checked: {i}")
        # backtrack and load beset path into trajectory object
        self.trajectory.addPoint(self.grid.grid_to_real(current))
        while current in traversed:
            current = traversed[current]
            if current is None:
                break
            self.trajectory.addPoint(self.grid.grid_to_real(current))
        self.trajectory.reverse()

        self.get_logger().info(f"A* planning time: {(self.get_clock().now() - start_time).nanoseconds / 1e9:.3f}s")

        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
