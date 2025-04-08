import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming SensorModel is defined in sensor_model.py
from sensor_model import SensorModel

class DummyNode:
    """ A dummy ROS 2 node to mimic ROS parameter handling for testing. """
    def declare_parameter(self, name, default_value):
        setattr(self, name, default_value)

    def get_parameter(self, name):
        class DummyParam:
            def __init__(self, value):
                self.value = value

            def get_parameter_value(self):
                return self

            @property
            def string_value(self):
                return self.value

            @property
            def integer_value(self):
                return self.value

            @property
            def double_value(self):
                return self.value

        return DummyParam(getattr(self, name))

    def get_logger(self):
        class Logger:
            def info(self, msg):
                print(msg)

        return Logger()

# Initialize a dummy node
node = DummyNode()

# Set reasonable parameters
node.map_topic = "test_map"
node.num_beams_per_particle = 180
node.scan_theta_discretization = 1.0
node.scan_field_of_view = np.pi
node.lidar_scale_to_map_scale = 10.0

# Create SensorModel instance
sensor_model = SensorModel(node)

# Precompute sensor model table
sensor_model.precompute_sensor_model()

# Create meshgrid for 3D plot
table_width = sensor_model.table_width
X, Y = np.meshgrid(np.arange(table_width), np.arange(table_width))
Z = sensor_model.sensor_model_table  # The probability values

# Create 3D figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')

# Labels and title
ax.set_xlabel("Ground Truth Distance (in px)")
ax.set_ylabel("Measured Distance (in px)")
ax.set_zlabel("P(Measured Distance | Ground Truth)")
ax.set_title("Precomputed Sensor Model")

plt.show()
