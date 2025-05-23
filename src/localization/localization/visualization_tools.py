from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker


class VisualizationTools:

    @staticmethod
    def plot_line(X, Y, publisher, color=(1.0, 0.0, 0.0), frame="/base_link"):
        """
        Publishes the points (x, y) to publisher
        so they can be visualized in rviz as
        connected line segments.
        Args:
            X, Y: The x and y values. These arrays
            must be of the same length.
            publisher: the publisher to publish to. The
            publisher must be of type Marker from the
            visualization_msgs.msg class.
            color: the RGB color of the plot.
            frame: the transformation frame to plot in.
        """
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
        for xi, yi in zip(X, Y):
            p = Point()
            p.x = xi
            p.y = yi
            line_strip.points.append(p)

        # Publish the line
        publisher.publish(line_strip)

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

    @staticmethod
    def plot_points(X, Y, publisher, color=(0.0, 1.0, 0.0), scale=0.1, frame="/base_link"):
        """
        Publishes individual points (x, y) to publisher so they can be visualized in rviz.
        Args:
            X, Y: The x and y values. These arrays must be of the same length.
            publisher: the publisher to publish to. The publisher must be of type Marker from the visualization_msgs.msg class.
            color: the RGB color of the points.
            scale: the scale of the points.
            frame: the transformation frame to plot in.
        """
        points_marker = Marker()
        points_marker.type = Marker.POINTS
        points_marker.header.frame_id = frame

        # Set the scale for the points (width and height)
        points_marker.scale.x = scale
        points_marker.scale.y = scale

        # Set the color for the points
        points_marker.color.a = 1.0
        points_marker.color.r = color[0]
        points_marker.color.g = color[1]
        points_marker.color.b = color[2]

        # Fill in the points
        for xi, yi in zip(X, Y):
            p = Point()
            p.x = xi
            p.y = yi
            p.z = 0.0
            points_marker.points.append(p)

        # Publish the points
        publisher.publish(points_marker)
