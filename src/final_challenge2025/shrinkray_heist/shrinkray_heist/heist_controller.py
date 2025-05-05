import math

from enum import Enum
from typing import Optional

from shrinkray_heist import traffic_light_detector

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import Point, PointStamped, Pose, PoseArray, PoseStamped, TransformStamped
from sensor_msgs.msg import Image

from cv_bridge import CvBridge

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

class State(Enum):
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

class HeistController(Node):

  def __init__(self):
    super().__init__("heist_controller")

    # self.active_pub = self.create_publisher(Bool, "/is_active", 1)
    self.points_sub = self.create_subscription(PoseArray, "/shell_points", self.points_callback, 1)
    self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
    self.at_goal_sub = self.create_subscription(Bool, "/at_goal", self.at_goal_callback, 1)
    self.drive_pub = self.create_publisher(AckermannDriveStamped, "/vesc/high_level/input/nav_0", 1)
    self.following_enable_pub = self.create_publisher(Bool, "/trajectory_following_enabled", 1)

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
    self.state: State = None
    self.next_state: Optional[State] = None

    self.poses: list = []
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

    self.seen_green = False


  def points_callback(self, pose_array: PoseArray) -> None:
    self.poses = list(reversed(list(map(copy_pose, pose_array.poses))))
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
      traffic_light_tf = self.tf_buffer.lookup_transform(
        "base_link",
        "traffic_light",
        Time()
      )
      dist = math.hypot(traffic_light_tf.transform.translation.x, traffic_light_tf.transform.translation.y)

      if dist < 2 and not self.seen_green:
        self.get_logger().info("NEXT TO TRAFFIC LIGHT")
        if not traffic_light_detector.light_is_green(self.image):
          self.get_logger().info("LIGHT IS NOT GREEN, DOING NOTHING")
          self.following_enable_pub.publish(Bool(data=False))
          msg = AckermannDriveStamped()
          msg.header.stamp = self.get_clock().now().to_msg()
          self.drive_pub.publish(msg)
          return
        self.seen_green = True
        self.following_enable_pub.publish(Bool(data=True))
    except TransformException as ex:
      pass


    # self.get_logger().info(f"{self.next_timestamp - self.get_clock().now().nanoseconds}")

    if self.state == State.WAIT_TIME:
      if self.get_clock().now().nanoseconds > self.next_timestamp:
        self.get_logger().info("wait 5s -> backup")
        self.state = State.BACKUP
        self.next_state = None
        self.next_timestamp = self.get_clock().now().nanoseconds + 5e9

    elif self.state == State.GOTO_POSE:
      if not self.poses:
        self.get_logger().info("FINISHED!")
        self.state = None
        self.next_state = None
        return

      self.goal_mut.pose = self.poses.pop()
      self.goal_pub.publish(self.goal_mut)
      self.state = State.WAIT_TRAJECTORY
      self.next_timestamp = self.get_clock().now().nanoseconds + 5e9
      self.next_state = State.GOTO_BANANA

    elif self.state == State.GOTO_BANANA:
      try:
        banana_tf = self.tf_buffer.lookup_transform(
          "map",
          "banana",
          Time()
        )
        base_link_tf = self.tf_buffer.lookup_transform(
          "map",
          "base_link",
          Time()
        )
        dx = banana_tf.transform.translation.x - base_link_tf.transform.translation.x
        dy = banana_tf.transform.translation.y - base_link_tf.transform.translation.y
        # normalize
        mag = 1 / math.hypot(dx, dy)
        dx *= mag
        dy *= mag

        self.goal_mut.pose.position.x = banana_tf.transform.translation.x - dx
        self.goal_mut.pose.position.y = banana_tf.transform.translation.y - dy

        self.get_logger().info(f"I SAW A BANANA!!!!!!!! {self.goal_mut.pose.position.x} {self.goal_mut.pose.position.y}")
        self.get_logger().info(f"STATE TRANSITION: GOTO_BANANA -> WAIT_TRAJECTORY (-> wait time) ")

        self.state = State.WAIT_TRAJECTORY
        self.next_state = State.WAIT_TIME

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
      msg = AckermannDriveStamped()
      msg.header.stamp = self.get_clock().now().to_msg()
      msg.drive.speed = -0.4
      self.following_enable_pub.publish(Bool(data=False))
      self.drive_pub.publish(msg)
      if self.get_clock().now().nanoseconds > self.next_timestamp:
        self.get_logger().info("backup -> goto next pose")
        self.state = State.GOTO_POSE
        self.next_state = None
        self.following_enable_pub.publish(Bool(data=True))
        self.next_timestamp = self.get_clock().now().nanoseconds + 5e9



def main(args=None):
    rclpy.init(args=args)
    controller = HeistController()
    rclpy.spin(controller)
    rclpy.shutdown()
