import copy
import math
from enum import Enum, auto
from typing import Optional

import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from cv_bridge import CvBridge
from geometry_msgs.msg import (
    Point,
    PointStamped,
    Pose,
    PoseArray,
    PoseStamped,
    PoseWithCovarianceStamped,
    TransformStamped,
)
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from shrinkray_heist import traffic_light_detector
from std_msgs.msg import Bool, String
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker

from . import visualize


class State(Enum):
    IDLE = auto()
    GOTO_POSE = auto()
    WAIT_FOR_PLAN_POINT = auto()
    WAIT_TRAJECTORY_POINT = auto()
    WAIT_FOR_PLAN_PARK = auto()
    WAIT_TRAJECTORY_PARK = auto()
    WAIT_PARKING_TIME = auto()
    WAIT_FOR_PLAN_BACKUP = auto()
    WAIT_TRAJECTORY_BACKUP = auto()
    WAIT_TRAJECTORY_BONUS = auto()


def copy_pose(incoming: Pose) -> Pose:
    pose = Pose()
    pose.position.x = incoming.position.x
    pose.position.y = incoming.position.y
    pose.position.z = incoming.position.z
    return pose


TRAFFIC_LIGHT: tuple[int, int] = (-12.3667, 14.6265)
CLOSENESS_POINT: tuple[int, int] = (-5.4, 22.7)


class HeistController(Node):

    def __init__(self):
        super().__init__("heist_controller")

        self.drive_topic = self.declare_parameter("drive_topic", "/vesc/high_level/input/nav_0").value
        self.main_map = self.declare_parameter("main_map", "").value
        self.bonus_map = self.declare_parameter("bonus_map", "").value
        self.is_sim = self.declare_parameter("is_sim", False).value
        if self.is_sim:
            self.get_logger().info("Running in simulation mode...")

        # self.active_pub = self.create_publisher(Bool, "/is_active", 1)
        self.points_sub = self.create_subscription(PoseArray, "/shell_points", self.points_callback, 1)
        self.goal_pub = self.create_publisher(PoseStamped, "/heist_goal_pose", 10)
        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self.initial_pose_callback, 1
        )
        self.at_goal_sub = self.create_subscription(Bool, "/at_goal", self.at_goal_callback, 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 1)
        self.following_enable_pub = self.create_publisher(Bool, "/trajectory_following_enabled", 1)
        self.following_backwards_pub = self.create_publisher(Bool, "/trajectory_following_backwards", 1)
        self.traffic_light_pub = self.create_publisher(Marker, "/traffic_light_point", 1)
        self.seen_banana_pub = self.create_publisher(Marker, "/detected_banana", 1)
        self.text_state_pub = self.create_publisher(Marker, "/heist_state", 1)
        self.update_map_pub = self.create_publisher(String, "/planner_update_map", 1)
        self.got_new_path_sub = self.create_subscription(Bool, "/got_new_path", self.got_new_path_callback, 1)

        self.odom_sub = self.create_subscription(
            Odometry, "/odom" if self.is_sim else "/pf/pose/odom", self.update_pose, 1
        )
        self.last_pose_mut: tuple[float, float] = (0, 0)

        self.watch_for_banana_pub = self.create_publisher(Bool, "/watch_for_banana", 1)

        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 1)
        self.bridge = CvBridge()
        # self.detector_sub = self.create_subscription(PoseStamped, "/detected_point", self 10)

        self.create_timer(0.1, self.update)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # self.active_mut = Bool()
        self.goal_mut = PoseStamped()

        self.next_timestamp = self.get_clock().now().nanoseconds

        # this is a terrible state machine
        # command based pls
        self.state: State = State.IDLE
        self.next_state: Optional[State] = None

        self.poses: list = []
        self.start_pose: Optional[Pose] = None
        self.last_point = None
        self.at_goal: bool = False
        self.got_new_path: bool = False

        # publish traffic light tf
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        traffic_light = TransformStamped()
        traffic_light.header.stamp = self.get_clock().now().to_msg()
        traffic_light.header.frame_id = "map"
        traffic_light.child_frame_id = "traffic_light"
        traffic_light.transform.translation.x = TRAFFIC_LIGHT[0]
        traffic_light.transform.translation.y = TRAFFIC_LIGHT[1]

        self.tf_static_broadcaster.sendTransform(traffic_light)

    def initial_pose_callback(self, pose: PoseWithCovarianceStamped) -> None:
        self.start_pose = copy_pose(pose.pose.pose)
        self.last_point = PoseStamped()
        self.last_point.pose = copy.deepcopy(self.start_pose)
        self.get_logger().info(f"RECEIVED INITIAL POSE: {self.start_pose}")

    def update_pose(self, odom: Odometry) -> None:
        self.last_pose_mut = (odom.pose.pose.position.x, odom.pose.pose.position.y)

    def points_callback(self, pose_array: PoseArray) -> None:
        self.poses = list(map(copy_pose, pose_array.poses))
        assert self.start_pose is not None, "Start pose was not set (2D pose estimate first!)"
        # Order by distance to start pose
        self.poses.sort(
            key=lambda pose: math.hypot(pose.position.x - CLOSENESS_POINT[0], pose.position.y - CLOSENESS_POINT[1]),
            reverse=True,
        )
        self.poses.insert(0, copy.deepcopy(self.start_pose))
        self.get_logger().info(f"received waypoints: {self.poses}")
        self.goal_mut.pose = copy.deepcopy(self.start_pose)
        self.state = State.GOTO_POSE

    def at_goal_callback(self, msg) -> None:
        self.get_logger().info("RECEIVED TRAJECTORY FINISH")
        self.at_goal = True

    def got_new_path_callback(self, msg) -> None:
        self.get_logger().info("RECEIVED NEW PATH")
        self.got_new_path = True

    def image_callback(self, msg: Image) -> None:
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def change_state(self, new_state: State, delay_sec: float = 1.5) -> None:
        self.get_logger().info(f"{self.state.name} -> {new_state.name}")
        self.state = new_state
        self.next_timestamp = self.get_clock().now().nanoseconds + delay_sec * 1e9

    def stop_moving(self) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.drive_pub.publish(msg)

    def update(self) -> None:
        within_banana_range = (
            math.hypot(
                self.last_pose_mut[0] - self.goal_mut.pose.position.x,
                self.last_pose_mut[1] - self.goal_mut.pose.position.y,
            )
            < 3.0
        )

        sees_banana = False
        try:
            banana_tf = self.tf_buffer.lookup_transform("map", "banana", Time())
            base_link_tf = self.tf_buffer.lookup_transform("map", "base_link", Time())
            base_link_tf.transform.translation.x += 0.3  # Drift bc still moving
            # Determine how old the banana transform is
            banana_timeout = 1.0
            banana_age: rclpy.duration.Duration = self.get_clock().now() - Time.from_msg(banana_tf.header.stamp)
            if banana_age.nanoseconds * 1e-9 > banana_timeout:
                raise TransformException("Banana transform is too old")
            visualize.plot_point(
                banana_tf.transform.translation.x,
                banana_tf.transform.translation.y,
                self.seen_banana_pub,
                scale=0.2,
                color=(1, 1, 0),
                frame="map",
            )
            sees_banana = True
        except TransformException as ex:
            pass

        # self.get_logger().info(f"{self.next_timestamp - self.get_clock().now().nanoseconds}")

        # State machine
        if self.state == State.IDLE:
            self.stop_moving()
            return

        elif self.state == State.GOTO_POSE:
            # Plan path once
            assert self.poses, "No poses to plan to"
            # Plan path to next point and wait
            self.last_point = copy.deepcopy(self.goal_mut)  # Save last goal point
            self.goal_mut.pose = self.poses.pop()  # Remove last point from list
            self.goal_pub.publish(copy.deepcopy(self.goal_mut))
            self.following_enable_pub.publish(Bool(data=False))
            self.following_backwards_pub.publish(Bool(data=False))
            self.stop_moving()
            self.change_state(State.WAIT_FOR_PLAN_POINT)

        elif self.state == State.WAIT_FOR_PLAN_POINT:
            # Wait for plan to finish
            self.stop_moving()
            if self.got_new_path:
                self.got_new_path = False
                # Follow trajectory and wait for it to finish
                self.following_enable_pub.publish(Bool(data=True))
                if self.poses:
                    self.change_state(State.WAIT_TRAJECTORY_POINT)  # Not the bonus point
                else:
                    self.change_state(State.WAIT_TRAJECTORY_BONUS)  # Bonus (last) point
                # Update map
                if len(self.poses) == 1:
                    self.update_map_pub.publish(String(data=self.bonus_map))  # Next plan uses bonus map
                else:
                    self.update_map_pub.publish(String(data=self.main_map))  # Next plan uses main map

        elif self.state == State.WAIT_TRAJECTORY_POINT:
            # Stop if we're at the goal
            if self.at_goal:
                self.at_goal = False
                # self.switch_state(State.LOOK_FOR_BANANA)
                # return

            # Look for red traffic light
            try:
                traffic_light_tf = self.tf_buffer.lookup_transform("base_link", "traffic_light", Time())
                dist = math.hypot(traffic_light_tf.transform.translation.x, traffic_light_tf.transform.translation.y)

                traffic_dist = 1.75
                if dist < traffic_dist:
                    if not self.is_sim:
                        is_red = traffic_light_detector.light_is_red(self.image)
                    else:
                        is_red = (self.get_clock().now().seconds_nanoseconds()[0] % 4) < 2
                    traffic_color = (1, 0, 0) if is_red else (0, 1, 0)
                    visualize.plot_point(
                        TRAFFIC_LIGHT[0],
                        TRAFFIC_LIGHT[1],
                        self.traffic_light_pub,
                        color=traffic_color,
                        scale=0.6,
                        frame="map",
                    )
                    if is_red:
                        # self.get_logger().info("Red light detected, stopping.")
                        self.following_enable_pub.publish(Bool(data=False))
                        self.stop_moving()
                        visualize.plot_debug_text("RED LIGHT", self.text_state_pub)
                        return
                    # self.get_logger().info("Near non-red light, continuing.")
                    self.following_enable_pub.publish(Bool(data=True))
                elif dist < traffic_dist + 0.5:
                    self.following_enable_pub.publish(Bool(data=True))
            except TransformException as ex:
                pass

            # Look for banana
            if within_banana_range and sees_banana:
                # Park in front of banana
                self.watch_for_banana_pub.publish(Bool(data=True))  # Signal to take an image
                dx = banana_tf.transform.translation.x - base_link_tf.transform.translation.x
                dy = banana_tf.transform.translation.y - base_link_tf.transform.translation.y
                mag = 1 / math.hypot(dx, dy)
                dx, dy = dx * mag, dy * mag
                parking_distance = 1.0
                self.goal_mut.pose.position.x = banana_tf.transform.translation.x - dx * parking_distance
                self.goal_mut.pose.position.y = banana_tf.transform.translation.y - dy * parking_distance
                self.goal_pub.publish(copy.deepcopy(self.goal_mut))
                self.following_enable_pub.publish(Bool(data=False))
                self.stop_moving()
                self.change_state(State.WAIT_FOR_PLAN_PARK)
            # self.get_logger().info(f"within_banana_range: {within_banana_range}, sees_banana: {sees_banana}")

        elif self.state == State.WAIT_FOR_PLAN_PARK:
            # Wait for plan to finish
            self.following_enable_pub.publish(Bool(data=False))
            self.stop_moving()
            if self.got_new_path:
                self.got_new_path = False
                # Follow trajectory and wait for it to finish
                self.following_enable_pub.publish(Bool(data=True))
                self.change_state(State.WAIT_TRAJECTORY_PARK)

        elif self.state == State.WAIT_TRAJECTORY_PARK:
            # Stop if we're at the goal
            if self.at_goal:
                self.at_goal = False
                self.following_enable_pub.publish(Bool(data=False))
                self.stop_moving()
                self.change_state(State.WAIT_PARKING_TIME, delay_sec=5.0)

        elif self.state == State.WAIT_PARKING_TIME:
            # Wait 5 seconds
            self.stop_moving()
            if self.get_clock().now().nanoseconds > self.next_timestamp:
                # Plan path to back up to last point
                self.goal_pub.publish(copy.deepcopy(self.last_point))
                self.following_enable_pub.publish(Bool(data=False))
                self.stop_moving()
                self.change_state(State.WAIT_FOR_PLAN_BACKUP)

        elif self.state == State.WAIT_FOR_PLAN_BACKUP:
            # Wait for plan to finish
            self.stop_moving()
            if self.got_new_path:
                self.got_new_path = False
                # Drive backwards slowly for 5 seconds
                self.following_backwards_pub.publish(Bool(data=True))
                self.following_enable_pub.publish(Bool(data=True))
                self.change_state(State.WAIT_TRAJECTORY_BACKUP, delay_sec=5.0)

        elif self.state == State.WAIT_TRAJECTORY_BACKUP:
            # Wait for 5 seconds (backup)
            if self.get_clock().now().nanoseconds > self.next_timestamp:
                self.following_enable_pub.publish(Bool(data=False))
                self.following_backwards_pub.publish(Bool(data=False))
                self.stop_moving()
                self.change_state(State.GOTO_POSE)  # Go to next pose

        elif self.state == State.WAIT_TRAJECTORY_BONUS:
            # Stop if we're at the goal
            if self.at_goal:
                self.at_goal = False
                self.following_enable_pub.publish(Bool(data=False))
                self.stop_moving()
                self.change_state(State.IDLE)

        visualize.plot_debug_text(self.state.name, self.text_state_pub)


def main(args=None):
    rclpy.init(args=args)
    controller = HeistController()
    rclpy.spin(controller)
    rclpy.shutdown()
