import rclpy
from rclpy.node import Node

class HeistController:
  def __init__(self, node: Node):
    self.node = node
    self.active_pub = self.create_publisher(bool, "/is_active", self.is_active, 1)
