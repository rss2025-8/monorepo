from enum import Enum
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Point, PointStamped, Pose, PoseArray, PoseStamped


class State(Enum):
  STOPPED = 0
  GOTO_FIRST_POINT = 1
  GOTO_SECOND_POINT = 2


class HeistController(Node):

  def __init__(self):
    super().__init__("heist_controller")

    self.active_pub = self.create_publisher(Bool, "/is_active", self.is_active, 1)
    self.points_sub = self.create_subscriber(PoseArray, "/shell_points", self.points_callback, 1)
    self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
    self.detector_sub = self.create_subscription(PoseStamped, "/detected_point", 10)
    self.state = State.GOTO_FIRST_POINT

    self.create_timer(0.01, self.update)

    self.start_timestamp = self.get_clock().now()
    self.next_state: Optional[State] = None

  def points_callback(self, pose_array: PoseArray) -> None:
    self.poses = pose_array.poses

  def update(self) -> None:
    if self.state == State.STOPPED:
      pass
    elif self.state == State.GOTO_FIRST_POINT:
      pass
    elif self.state == State.GOTO_SECOND_POINT:
      pass

def main(args=None):
    rclpy.init(args=args)
    controller = HeistController()
    rclpy.spin(controller)
    rclpy.shutdown()
