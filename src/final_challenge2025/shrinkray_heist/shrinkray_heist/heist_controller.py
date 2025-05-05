from enum import Enum
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool
from geometry_msgs.msg import Point, PointStamped, Pose, PoseArray, PoseStamped

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException


class State(Enum):
  WAIT_TIME = 0
  WAIT_TRAJECTORY = 1
  GOTO_POSE = 2
  GOTO_BANANA = 3


def copy_pose(incoming: Pose) -> Pose:
  pose = Pose()
  pose.position.x = incoming.position.x
  pose.position.y = incoming.position.y
  pose.position.z = incoming.position.z
  return pose


class HeistController(Node):

  def __init__(self):
    super().__init__("heist_controller")

    # self.active_pub = self.create_publisher(Bool, "/is_active", 1)
    self.points_sub = self.create_subscription(PoseArray, "/shell_points", self.points_callback, 1)
    self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
    self.at_goal_sub = self.create_subscription(Bool, "/at_goal", self.at_goal_callback, 1)
    # self.detector_sub = self.create_subscription(PoseStamped, "/detected_point", self 10)

    self.create_timer(0.2, self.update)

    self.tf_buffer = Buffer()
    self.tf_listener = TransformListener(self.tf_buffer, self)

    # self.active_mut = Bool()
    self.goal_mut = PoseStamped()

    self.next_timestamp = self.get_clock().now()

    # this is a terrible state machine
    # command based pls
    self.state: State = None
    self.next_state: Optional[State] = None

    self.poses: list = []
    self.at_goal: bool = False

  def points_callback(self, pose_array: PoseArray) -> None:
    self.poses = list(reversed(list(map(copy_pose, pose_array.poses))))
    self.get_logger().info(f"received two waypoints: {self.poses}")
    self.state = State.GOTO_POSE

  def at_goal_callback(self, idk) -> None:
    self.get_logger().info("RECEIVED TRAJECTORY FINISH")
    self.at_goal = True

  # def detection_callback(self, detected_pose: )

  def update(self) -> None:
    if self.state == State.WAIT_TIME:
      if self.get_clock().now() > self.next_timestamp:
        self.state = self.next_state

    elif self.state == State.GOTO_POSE:
      if not self.poses:
        self.get_logger().info("FINISHED!")
        self.state = None
        return

      self.goal_mut.pose = self.poses.pop()
      self.goal_pub.publish(self.goal_mut)
      self.state = State.WAIT_TRAJECTORY
      self.next_state = State.GOTO_BANANA

    elif self.state == State.GOTO_BANANA:
      try:
        banana_tf = self.tf_buffer.lookup_transform(
          "map",
          "banana",
          Time()
        )
        self.goal_mut.pose.position.x = banana_tf.transform.translation.x
        self.goal_mut.pose.position.y = banana_tf.transform.translation.y

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
        self.get_logger().info("FINISHED TRAJECTORY, MOVING ON TO NEXT STATE")
        self.at_goal = False
        self.next_timestamp = self.get_clock().now().nanoseconds / 10e9
        self.state = self.next_state
        self.next_state = State.GOTO_POSE

def main(args=None):
    rclpy.init(args=args)
    controller = HeistController()
    rclpy.spin(controller)
    rclpy.shutdown()
