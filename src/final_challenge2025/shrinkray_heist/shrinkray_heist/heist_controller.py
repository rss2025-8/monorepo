import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

class HeistController:
  def __init__(self, node: Node):
    self.node = node
    self.active_pub = self.create_publisher(Bool, "/is_active", self.is_active, 1)
