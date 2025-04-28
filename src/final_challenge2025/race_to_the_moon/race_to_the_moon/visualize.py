import math

import numpy as np
import race_to_the_moon.homography as homography
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


def plot_line(X, Y, publisher, color=(1.0, 0.0, 0.0), scale=0.1, z=0.0, frame="/base_link"):
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
        scale: the scale of the line.
        z: the z coordinate of the line.
        frame: the transformation frame to plot in.
    """
    if publisher.get_subscription_count() == 0:
        return
    # Construct a line
    line_strip = Marker()
    line_strip.type = Marker.LINE_STRIP
    line_strip.header.frame_id = frame

    # Set the size and color
    line_strip.scale.x = scale
    line_strip.scale.y = scale
    line_strip.color.a = 1.0
    line_strip.color.r = float(color[0])
    line_strip.color.g = float(color[1])
    line_strip.color.b = float(color[2])

    # Fill the line with the desired values
    for xi, yi in zip(X, Y):
        p = Point()
        p.x = xi
        p.y = yi
        p.z = z
        line_strip.points.append(p)

    # Publish the line
    publisher.publish(line_strip)


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
    if publisher.get_subscription_count() == 0:
        return
    text_marker = Marker()
    text_marker.type = Marker.TEXT_VIEW_FACING
    text_marker.header.frame_id = frame
    text_marker.text = text
    text_marker.pose.position.x = x
    text_marker.pose.position.y = y
    text_marker.pose.position.z = z
    text_marker.scale.z = scale
    text_marker.scale.x = scale / 1.5
    text_marker.color.a = 1.0
    text_marker.color.r = float(color[0])
    text_marker.color.g = float(color[1])
    text_marker.color.b = float(color[2])
    publisher.publish(text_marker)


def plot_debug_text(text, publisher, color=(1.0, 0.0, 0.0)):
    """Plots debug text on top of the car. Clear the debug text by calling `clear_marker(publisher)`."""
    plot_text(text, 0.0, 0.0, 0.5, 0.3, publisher, color)


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
    if publisher.get_subscription_count() == 0:
        return
    points_marker = Marker()
    points_marker.type = Marker.POINTS
    points_marker.header.frame_id = frame

    # Set the scale for the points (width and height)
    points_marker.scale.x = scale
    points_marker.scale.y = scale

    # Set the color for the points
    points_marker.color.a = 1.0
    points_marker.color.r = float(color[0])
    points_marker.color.g = float(color[1])
    points_marker.color.b = float(color[2])

    # Fill in the points
    for xi, yi in zip(X, Y):
        p = Point()
        p.x = xi
        p.y = yi
        p.z = 0.0
        points_marker.points.append(p)

    # Publish the points
    publisher.publish(points_marker)


def plot_point(x, y, publisher, color=(0.0, 1.0, 0.0), scale=0.3, frame="/base_link"):
    """
    Publishes a single point (x, y) as a sphere to publisher for visualization in rviz.
    """
    if publisher.get_subscription_count() == 0:
        return
    point_marker = Marker()
    point_marker.type = Marker.SPHERE
    point_marker.header.frame_id = frame
    point_marker.pose.position.x = x
    point_marker.pose.position.y = y
    point_marker.pose.position.z = 0.0
    point_marker.scale.x = scale
    point_marker.scale.y = scale
    point_marker.scale.z = scale
    point_marker.color.a = 1.0
    point_marker.color.r = float(color[0])
    point_marker.color.g = float(color[1])
    point_marker.color.b = float(color[2])
    publisher.publish(point_marker)


def plot_circle(x, y, radius, publisher, color=(0.0, 1.0, 0.0), scale=0.1, z=0.0, frame="/base_link"):
    """
    Publishes a circle (outlined) to publisher for visualization in rviz.

    Args:
        x, y: the center of the circle.
        radius: the radius of the circle.
        publisher: the publisher to publish to.
        color: the color of the circle.
        scale: the scale of the circle.
        z: the z coordinate of the circle.
        frame: the transformation frame to plot in.
    """
    if publisher.get_subscription_count() == 0:
        return
    circle_marker = Marker()
    circle_marker.type = Marker.LINE_STRIP
    circle_marker.header.frame_id = frame
    circle_marker.scale.x = scale
    circle_marker.color.a = 1.0
    circle_marker.color.r = float(color[0])
    circle_marker.color.g = float(color[1])
    circle_marker.color.b = float(color[2])
    num_points = 18 + int(abs(radius))
    for i in range(num_points):
        p = Point()
        p.x = x + radius * math.cos(i * 2 * math.pi / (num_points - 1))
        p.y = y + radius * math.sin(i * 2 * math.pi / (num_points - 1))
        p.z = z
        circle_marker.points.append(p)
    publisher.publish(circle_marker)


def clear_marker(publisher, frame="/base_link"):
    """Clears the marker by publishing an empty marker. Must pass the frame that was used (defaults to /base_link)."""
    if publisher.get_subscription_count() == 0:
        return
    marker = Marker()
    marker.action = Marker.DELETEALL
    marker.header.frame_id = frame
    publisher.publish(marker)


precomputed_xy, precomputed_uv = None, None


def plot_image(image, publisher, scale=0.05, sample_shape=None, frame="/base_link"):
    """
    Uses homography to project a camera image onto the ground as a PointCloud2, for visualization in rviz.
    Assumes the same parameters are used for every call (cached).

    Args:
        image: H*W*3 uint8 BGR OpenCV image
        publisher: the publisher to publish to.
        scale: the scale of the points.
        sample_shape: (sample_height, sample_width) tuple of number of samples along image height and width
        frame: the transformation frame to plot in.
    """
    if publisher.get_subscription_count() == 0:
        return

    h, w, _ = image.shape
    marker = Marker()
    marker.type = Marker.POINTS
    marker.header.frame_id = frame
    marker.scale.x = scale
    marker.scale.y = scale
    marker.color.a = 1.0

    global precomputed_xy, precomputed_uv
    if precomputed_xy is None:
        # Precompute (x, y) and integer (u, v) coordinates
        x_coords = np.linspace(0, 5, 72, dtype=float)
        y_coords = np.linspace(-2, 2, 64, dtype=float)
        precomputed_xy, precomputed_uv = [], []
        for x in x_coords:
            for y in y_coords:
                u, v = homography.transform_xy_to_uv(x, y)
                u_int, v_int = round(u), round(v)
                if u_int < 0 or u_int >= w or v_int < 0 or v_int >= h:
                    continue
                precomputed_xy.append((x, y))
                precomputed_uv.append((u_int, v_int))
        precomputed_xy = np.array(precomputed_xy)
        precomputed_uv = np.array(precomputed_uv)

    # Get image colors
    colors = image[precomputed_uv[:, 1], precomputed_uv[:, 0]]
    marker.points = [Point(x=x, y=y, z=0.0) for x, y in precomputed_xy]
    marker.colors = [ColorRGBA(r=r / 255.0, g=g / 255.0, b=b / 255.0, a=1.0) for b, g, r in colors]
    publisher.publish(marker)
