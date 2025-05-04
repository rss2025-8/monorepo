# CTE is the distance from (0, 0) to the trajectory / mid line (they are equivalent just different msg forms)

# mid line topic is /race/mid_lane and is of type Marker

import math
import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import rclpy
import tf_transformations
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.interpolate import splev, splprep
from std_msgs.msg import Float32, Header
from visualization_msgs.msg import Marker
from vs_msgs.msg import Point, Trajectory

from . import visualize


class CTE(Node):
    def __init__(self):
        super().__init__("cte_node")
        self.declare_parameter("cte_topic", "/cte")
        self.declare_parameter("mid_lane_topic", "/race/mid_lane")

        self.mid_lane_sub = self.create_subscription(
            Marker,
            self.get_parameter("mid_lane_topic").get_parameter_value().string_value,
            self.mid_lane_callback,
            10,
        )
        self.cte_pub = self.create_publisher(
            Float32,
            self.get_parameter("cte_topic").get_parameter_value().string_value,
            10,
        )
    
    def mid_lane_callback(self, msg: Marker):
        # Extract pose from the mid lane marker
        points = msg.points
        start_point = points[0]
        end_point = points[-1]

        # find distance from (0,0) to the line formed by the start and end points
        x1 = start_point.x
        y1 = start_point.y
        x2 = end_point.x
        y2 = end_point.y
        # calculate the slope (m) and y-intercept (b)
        if x2 - x1 == 0:
            m = float("inf")
            b = float("inf")
        else:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
        # calculate the distance from (0, 0) to the line
        # distance = |Ax + By + C| / sqrt(A^2 + B^2)
        # where A = -m, B = 1, C = -b
        A = -m
        B = 1
        C = -b
        distance = abs(C) / math.sqrt(A**2 + B**2)
        

        # publish the distance
        distance_msg = Float32()
        distance_msg.data = distance
        self.cte_pub.publish(distance_msg)

def main(args=None):

    # profiler.enable()

    rclpy.init(args=args)

    follower = CTE()

    rclpy.spin(follower)

    rclpy.shutdown()








