from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
import numpy as np
from rclpy.duration import Duration


class VisualizationTools:

    @staticmethod
    def plot_line(x, y, publisher, color=(1.0, 0.0, 0.0), frame="/base_link"):
        """
        Publishes the points (x, y) to publisher
        so they can be visualized in rviz as
        connected line segments.
        Args:
            x, y: The x and y values. These arrays
            must be of the same length.
            publisher: the publisher to publish to. The
            publisher must be of type Marker from the
            visualization_msgs.msg class.
            color: the RGB color of the plot.
            frame: the transformation frame to plot in.
        """

        assert isinstance(x, (list, tuple, np.ndarray)), "x must be a list, tuple, or numpy array"
        assert isinstance(y, (list, tuple, np.ndarray)), "y must be a list, tuple, or numpy array"
        assert len(x) == len(y), "x and y must have the same length"
        assert len(x) > 0, "x and y must not be empty"
        assert hasattr(publisher, 'publish'), "publisher must have a publish method"
        assert callable(publisher.publish), "publisher.publish must be callable"
        assert isinstance(color, (list, tuple)) and len(color) == 3, "color must be a tuple or list of 3 RGB values"
        assert all(0 <= c <= 1.0 for c in color), "color values must be between 0 and 1"
        assert isinstance(frame, str), "frame must be a string"

        # Construct a line
        line_strip = Marker()
        line_strip.type = Marker.LINE_STRIP
        line_strip.header.frame_id = frame

        # Set the size and color
        line_strip.scale.x = 0.1
        line_strip.scale.y = 0.1
        line_strip.color.a = 1.0
        line_strip.color.r = color[0]
        line_strip.color.g = color[1]
        line_strip.color.b = color[2]

        # Fill the line with the desired values
        for xi, yi in zip(x, y):
            p = Point()
            p.x = xi
            p.y = yi
            line_strip.points.append(p)

        # Publish the line
        publisher.publish(line_strip)

    @staticmethod
    def plot_point(x, y, publisher, color=(1.0, 0.0, 0.0), frame="/base_link", lifetime=1.0):
        """
        Publishes the point (x, y) to publisher
        so they can be visualized in rviz.
        Args:
            x, y: The x and y values. These arrays
            must be of the same length.
            publisher: the publisher to publish to. The
            publisher must be of type Marker from the
            visualization_msgs.msg class.
            color: the RGB color of the plot.
            frame: the transformation frame to plot in.
        """

        assert isinstance(x, (int, float)), f"x must be a number, got {type(x)}"
        assert isinstance(y, (int, float)), f"y must be a number, got {type(y)}"
        assert hasattr(publisher, 'publish'), "publisher must have a publish method"
        assert callable(publisher.publish), "publisher.publish must be callable"
        assert isinstance(color, (list, tuple)) and len(color) == 3, "color must be a tuple or list of 3 RGB values"
        assert all(0 <= c <= 1.0 for c in color), "color values must be between 0 and 1"
        assert isinstance(frame, str), "frame must be a string"
        assert isinstance(lifetime, (int, float)) and lifetime > 0, "lifetime must be a positive number"

        # Construct a point marker
        point_marker = Marker()
        point_marker.type = Marker.POINTS
        point_marker.header.frame_id = frame
        point_marker.action = Marker.ADD

        # Set the size and color
        point_marker.scale.x = 0.2  # Point size
        point_marker.scale.y = 0.2  # Point size
        point_marker.color.a = 1.0  # Full opacity
        point_marker.color.r = color[0]
        point_marker.color.g = color[1]
        point_marker.color.b = color[2]

        point_marker.lifetime = rclpy.duration.Duration(seconds=lifetime).to_msg()

        # Fill the line with the desired values
        p = Point()
        p.x = x
        p.y = y
        p.z = 0.0
        point_marker.points.append(p)

        # Publish the line
        publisher.publish(point_marker)

    @staticmethod
    def plot_text(text, x, y, z, scale, publisher, color=(1.0, 0.0, 0.0), frame="/base_link"):
        """
        Publishes the given text to publisher for visualization in rviz.
        Args:
            text: The text to show.
            x, y, z: The x, y, and z location of the text.
            scale: The scale of the text.
            publisher: the publisher to publish to. The
            publisher must be of type Marker from the
            visualization_msgs.msg class.
            color: the RGB color of the plot.
            frame: the transformation frame to plot in.
        """

        assert isinstance(text, str), f"text must be a string, got {type(text)}"
        assert isinstance(x, (int, float)), f"x must be a number, got {type(x)}"
        assert isinstance(y, (int, float)), f"y must be a number, got {type(y)}"
        assert isinstance(z, (int, float)), f"z must be a number, got {type(z)}"
        assert isinstance(scale, (int, float)) and scale > 0, f"scale must be a positive number, got {scale}"
        assert hasattr(publisher, 'publish'), "publisher must have a publish method"
        assert callable(publisher.publish), "publisher.publish must be callable"
        assert isinstance(color, (list, tuple)) and len(color) == 3, "color must be a tuple or list of 3 RGB values"
        assert all(0 <= c <= 1.0 for c in color), "color values must be between 0 and 1"
        assert isinstance(frame, str), "frame must be a string"


        text_marker = Marker()
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.header.frame_id = frame
        text_marker.text = text
        text_marker.pose.position.x = x
        text_marker.pose.position.y = y
        text_marker.pose.position.z = z
        text_marker.scale.z = scale
        text_marker.color.a = 1.0
        text_marker.color.r = color[0]
        text_marker.color.g = color[1]
        text_marker.color.b = color[2]
        publisher.publish(text_marker)
