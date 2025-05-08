import copy
import math
from enum import Enum
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
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from shrinkray_heist import traffic_light_detector
from std_msgs.msg import Bool
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker

from . import visualize


class State(Enum):
    NONE = -1
    WAIT_TIME = 0
    WAIT_TRAJECTORY = 1
    GOTO_POSE = 2
    GOTO_BANANA = 3
    BACKUP = 4


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

        self.odom_sub = self.create_subscription(Odometry, "/odom", self.update_pose, 1)
        self.last_pose_mut: tuple[float, float] = (0, 0)

        self.banana_state_pub = self.create_publisher(Bool, "/i_hate_ros", 1)

        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 1)
        self.bridge = CvBridge()
        # self.detector_sub = self.create_subscription(PoseStamped, "/detected_point", self 10)

        self.create_timer(0.2, self.update)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # self.active_mut = Bool()
        self.goal_mut = PoseStamped()

        self.next_timestamp = self.get_clock().now().nanoseconds

        # this is a terrible state machine
        # command based pls
        self.state: State = State.NONE
        self.next_state: Optional[State] = None

        self.poses: list = []
        self.start_pose: Optional[Pose] = None
        self.last_point_pose: Optional[Pose] = None
        self.at_goal: bool = False

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
        self.get_logger().info(f"RECEIVED INITIAL POSE: {self.start_pose}")

    def update_pose(self, odom: Odometry) -> None:
        self.last_pose_mut = (odom.pose.pose.position.x, odom.pose.pose.position.y)

    def points_callback(self, pose_array: PoseArray) -> None:
        self.poses = list(map(copy_pose, pose_array.poses))
        assert self.start_pose is not None
        # Order by distance to start pose
        self.poses.sort(
            key=lambda pose: math.hypot(
                pose.position.x - CLOSENESS_POINT[0], pose.position.y - CLOSENESS_POINT[1]
            ),
            reverse=True
        )
        self.get_logger().info(f"received two waypoints: {self.poses}")
        self.state = State.GOTO_POSE
        self.next_state = None

    def at_goal_callback(self, idk) -> None:
        self.get_logger().info("RECEIVED TRAJECTORY FINISH")
        self.at_goal = True

    def image_callback(self, msg: Image) -> None:
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def update(self) -> None:
        try:
            traffic_light_tf = self.tf_buffer.lookup_transform("base_link", "traffic_light", Time())
            dist = math.hypot(traffic_light_tf.transform.translation.x, traffic_light_tf.transform.translation.y)

            if dist < 2:
                self.get_logger().info("NEXT TO TRAFFIC LIGHT")
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
                    self.get_logger().info("LIGHT IS RED, DOING NOTHING")
                    self.following_enable_pub.publish(Bool(data=False))
                    msg = AckermannDriveStamped()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    self.drive_pub.publish(msg)
                    visualize.plot_debug_text("RED LIGHT", self.text_state_pub)
                    return
                self.following_enable_pub.publish(Bool(data=True))
        except TransformException as ex:
            pass

        within_banana_range = math.hypot(self.last_pose_mut[0] - self.goal_mut.pose.position.x, self.last_pose_mut[1] - self.goal_mut.pose.position.y) < 3

        # self.get_logger().info(f"{self.next_timestamp - self.get_clock().now().nanoseconds}")

        if self.state == State.WAIT_TIME:
            if self.get_clock().now().nanoseconds > self.next_timestamp:
                self.get_logger().info("wait 5s -> backup")
                # Plan path to last point
                self.get_logger().info(f"BACKING UP TO {self.last_point}")
                self.goal_pub.publish(self.last_point)
                self.following_backwards_pub.publish(Bool(data=True))
                self.state = State.BACKUP
                self.next_state = None
                self.next_timestamp = self.get_clock().now().nanoseconds + 5e9

        elif self.state == State.GOTO_POSE:
            if not self.poses:
                if self.start_pose is not None:
                    self.get_logger().info("Going back to start!")
                    self.last_point = copy.deepcopy(self.goal_mut)
                    self.goal_mut.pose = self.start_pose
                    self.start_pose = None  # So we don't do it again
                    self.goal_pub.publish(self.goal_mut)
                    self.state = State.WAIT_TRAJECTORY
                    self.next_state = State.GOTO_POSE
                else:
                    self.get_logger().info("Finished!")
                    self.state = State.NONE
                return

            self.last_point = copy.deepcopy(self.goal_mut)
            self.goal_mut.pose = self.poses.pop()
            self.goal_pub.publish(self.goal_mut)
            self.state = State.WAIT_TRAJECTORY
            self.next_timestamp = self.get_clock().now().nanoseconds + 8e9
            self.next_state = State.GOTO_BANANA
            self.banana_state_pub.publish(Bool(data=True))

            # no longer seeing banana
            # self.banana_state_pub.publish(Bool(data=False))

        elif self.state == State.GOTO_BANANA or (self.state == State.WAIT_TRAJECTORY and within_banana_range and self.get_clock().now().nanoseconds > self.next_timestamp):
            try:
                banana_tf = self.tf_buffer.lookup_transform("map", "banana", Time())
                base_link_tf = self.tf_buffer.lookup_transform("map", "base_link", Time())
                visualize.plot_point(
                    banana_tf.transform.translation.x,
                    banana_tf.transform.translation.y,
                    self.seen_banana_pub,
                    scale=0.2,
                    color=(1, 1, 0),
                    frame="map",
                )
                dx = banana_tf.transform.translation.x - base_link_tf.transform.translation.x
                dy = banana_tf.transform.translation.y - base_link_tf.transform.translation.y
                # normalize
                mag = 1 / math.hypot(dx, dy)
                dx *= mag
                dy *= mag

                parking_distance = 0.5
                self.goal_mut.pose.position.x = banana_tf.transform.translation.x - dx * parking_distance
                self.goal_mut.pose.position.y = banana_tf.transform.translation.y - dy * parking_distance

                self.get_logger().info(
                    f"I SAW A BANANA!!!!!!!! {self.goal_mut.pose.position.x} {self.goal_mut.pose.position.y}"
                )
                self.get_logger().info(f"STATE TRANSITION: GOTO_BANANA -> WAIT_TRAJECTORY (-> wait time) ")

                self.state = State.WAIT_TRAJECTORY
                self.next_state = State.WAIT_TIME
                self.next_timestamp = float('inf')

                self.goal_pub.publish(self.goal_mut)
            except TransformException as ex:
                pass
                # self.get_logger().info(
                #     f'Could not transform map to banana: {ex}')

        elif self.state == State.WAIT_TRAJECTORY:
            if self.at_goal:
                self.get_logger().info(f"follow trajectory -> {self.next_state.name}")
                self.at_goal = False
                self.next_timestamp = self.get_clock().now().nanoseconds + 5e9
                self.state = self.next_state
                self.next_state = None
                # self.next_state = State.WAIT_TIME

        elif self.state == State.BACKUP:
            # msg = AckermannDriveStamped()
            # msg.header.stamp = self.get_clock().now().to_msg()
            # msg.drive.speed = -0.4
            # self.following_enable_pub.publish(Bool(data=False))
            # self.drive_pub.publish(msg)
            if self.get_clock().now().nanoseconds > self.next_timestamp:
                self.get_logger().info("backup -> goto next pose")
                self.state = State.GOTO_POSE
                self.next_state = None
                self.following_backwards_pub.publish(Bool(data=False))
                # self.following_enable_pub.publish(Bool(data=True))
                self.next_timestamp = self.get_clock().now().nanoseconds + 5e9

        visualize.plot_debug_text(self.state.name, self.text_state_pub)


def main(args=None):
    rclpy.init(args=args)
    controller = HeistController()
    rclpy.spin(controller)
    rclpy.shutdown()
