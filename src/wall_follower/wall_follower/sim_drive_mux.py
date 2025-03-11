import rclpy

# ackermann message
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node


class SimDriveMux(Node):

    def __init__(self):
        super().__init__("sim_drive_mux")
        self.declare_parameter("safety_controller_topic", "default")
        self.declare_parameter("wall_follower_topic", "default")
        self.declare_parameter("drive_topic", "default")

        self.subscription_safety = self.create_subscription(
            AckermannDriveStamped,
            self.get_parameter("safety_controller_topic").get_parameter_value().string_value,
            self.safety_callback,
            10,
        )
        self.subscription_wall = self.create_subscription(
            AckermannDriveStamped,
            self.get_parameter("wall_follower_topic").get_parameter_value().string_value,
            self.wall_callback,
            10,
        )
        self.publisher_drive = self.create_publisher(
            AckermannDriveStamped,
            self.get_parameter("drive_topic").get_parameter_value().string_value,
            10,
        )
        self.unsafe_detection = False

    def safety_callback(self, msg):
        self.get_logger().info("SAFETY CONTROLLER FOUND UNSAFE SITUATION")
        self.unsafe_detection = True
        # Command drive topic to stop
        self.command_drive(0.0, 0.0)

    def wall_callback(self, msg):
        if not self.unsafe_detection:
            self.get_logger().info("WALL FOLLOWER CONTROLLER")
            # Command drive topic to follow wall
            self.command_drive(msg.drive.speed, msg.drive.steering_angle)
        else:
            self.get_logger().info("OVERRIDING WALL FOLLOWER, STOPPING")
            # Command drive topic to stop
            self.command_drive(0.0, 0.0)

    def command_drive(self, speed, angle):
        msg = AckermannDriveStamped()
        msg.drive.speed = speed
        msg.drive.steering_angle = angle
        self.publisher_drive.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    sim_drive_mux = SimDriveMux()

    rclpy.spin(sim_drive_mux)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    sim_drive_mux.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
