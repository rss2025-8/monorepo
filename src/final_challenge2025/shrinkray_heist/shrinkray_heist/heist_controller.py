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
  STOPPED = 0
  GOTO_FIRST_POINT = 1
  GOTO_SECOND_POINT = 2
  GOTO_BANANA = 3
  GOINGTO_BANANA = 4


class HeistController(Node):

  def __init__(self):
    super().__init__("heist_controller")

    self.active_pub = self.create_publisher(Bool, "/is_active", 1)
    self.points_sub = self.create_subscription(PoseArray, "/shell_points", self.points_callback, 1)
    self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
    # self.detector_sub = self.create_subscription(PoseStamped, "/detected_point", self 10)
    self.state = State.GOTO_BANANA

    self.create_timer(0.2, self.update)

    self.tf_buffer = Buffer()
    self.tf_listener = TransformListener(self.tf_buffer, self)

    self.active_mut = Bool()
    self.goal_mut = PoseStamped()

    self.next_timestamp = self.get_clock().now()
    self.next_state: Optional[zed_camera_frameState] = None

  def points_callback(self, pose_array: PoseArray) -> None:
    self.poses = pose_array.poses

  # def detection_callback(self, detected_pose: )

  def update(self) -> None:
    self.active_mut.data = False

    if self.state == State.STOPPED:
      # if self.get_clock().now() > self.next_timestamp:
      #   self.state = self.next_state
      pass
    elif self.state == State.GOTO_FIRST_POINT:
      pass
    elif self.state == State.GOTO_SECOND_POINT:
      pass
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

        self.active_mut.data = True

        self.state = State.GOINGTO_BANANA

        self.goal_pub.publish(self.goal_mut)
      except TransformException as ex:
        pass
        # self.get_logger().info(
        #     f'Could not transform map to banana: {ex}')

    elif self.state == State.GOINGTO_BANANA:
      pass


    self.active_pub.publish(self.active_mut)

def main(args=None):
    rclpy.init(args=args)
    controller = HeistController()
    rclpy.spin(controller)
    rclpy.shutdown()
