import rclpy
from rclpy.node import Node

# import dubins
import cv2
from scipy.ndimage import gaussian_filter

import math
from typing import Iterable
from dataclasses import dataclass
import numpy as np
import queue

assert rclpy
from std_msgs.msg import Header
import geometry_msgs
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose, Quaternion
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory


def convolution_downsample(mat: np.ndarray, kernel_size: int):
    '''idk what a "numpy" is
    fortunately this is only ran once'''

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
        inner_blur = gaussian_filter(downsampled_grid, sigma=3)
        outer_blur = gaussian_filter(downsampled_grid, sigma=9)
        # smooth = 

        return cls(
            grid=np.maximum(np.maximum(inner_blur, outer_blur), downsampled_grid),
            resolution=msg.info.resolution * downsampling,
            origin=Point(msg.info.origin.position.x, msg.info.origin.position.y)
        )

    def get_weight(self, point: Point) -> float:
        grid_y = int(point.y)
        grid_x = int(point.x)

        if self.in_bounds(grid_x, grid_y):
            return self.grid[grid_y, grid_x]

        return float('+inf')

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
        return PoseArray(header=Header(frame_id="map"), poses=[
                             Pose(position=geometry_msgs.msg.Point(y=-x*resolution+self.origin.y, x=self.origin.x-y*resolution, z=0.0)) for x, y in free_coords[::10]
                         ])


def dubins_dist(start: Point, end: Point) -> float:
    pass


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.debug_pub = self.create_publisher(
            PoseArray,
            "/debug_map",
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")
        self.last_pose: Pose
        self.grid: Grid = Grid.empty()

    def map_cb(self, msg: OccupancyGrid):
        self.grid = Grid.from_msg(msg, downsampling=7)
        self.get_logger().info(f"orientation {msg.info.origin.orientation}")
        self.get_logger().info(f"Received grid: {self.grid}; shape: {self.grid.grid.shape}; neighbors {list(self.grid.get_weight(v) for v in self.grid.get_neighbors(Point(0, 0)))}")
        self.debug_pub.publish(self.grid.get_free_arr(self.grid.resolution))

    def pose_cb(self, pose: PoseWithCovarianceStamped):
        self.last_pose = Point.from_msg(pose.pose.pose.position, self.grid)
        self.get_logger().info(f"Received pose: {self.last_pose}; ({pose.pose.pose.position.x}, {pose.pose.pose.position.y})")

    def goal_cb(self, msg: PoseStamped):
        self.plan_path(self.last_pose, Point.from_msg(msg.pose.position, self.grid), self.grid)

    def plan_path(self, start_point: Point, end_point: Point, map: Grid):
        self.trajectory.clear()

        agenda = queue.PriorityQueue()
        agenda.put((0.0, start_point))
        traversed = {start_point: None}
        cost_map = {start_point: 0.0}

        while not agenda.empty():
            current: Point = agenda.get()[1]
            self.get_logger().info(f"current: {current} | nbrs: {list(map.get_weight(v) for v in  map.get_neighbors(current))}")

            if current == end_point:
                break

            for next in map.get_neighbors(current):
                weight = map.get_weight(next)
                if weight > 95:
                    continue
                self.get_logger().info(f"next point: {next.as_tuple()}")
                new_cost = cost_map[current] + 1 + weight

                if next not in cost_map or new_cost < cost_map[next]:
                    cost_map[next] = new_cost

                    priority = new_cost + next.dist(end_point)
                    self.get_logger().info(f"priority: {priority}")
                    agenda.put((priority, next))
                    traversed[next] = current

        # backtrack and load beset path into trajectory object
        self.trajectory.addPoint(self.grid.grid_to_real(current))
        while current in traversed:
            current = traversed[current]
            if current is None:
                break
            self.trajectory.addPoint(self.grid.grid_to_real(current))
            self.trajectory.reverse()

        self.get_logger().info("FINISHED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.get_logger().info(f"trajectory: {self.trajectory.points}")

        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
