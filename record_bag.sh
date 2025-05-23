#!/usr/bin/env bash

# Core topics
TOPICS="/map /robot_description /tf /tf_static /pf/pose/odom"
# Sensors
# TOPICS="$TOPICS /scan /vesc/odom"
# Path planner visuals
# TOPICS="$TOPICS /followed_trajectory/start_point /followed_trajectory/end_pose /followed_trajectory/path /planned_trajectory/start_point /planned_trajectory/end_pose /planned_trajectory/path"
# Pure pursuit visuals
TOPICS="$TOPICS /pure_pursuit/drive_line /pure_pursuit/driving_arc /pure_pursuit/lookahead_circle /pure_pursuit/lookahead_point /pure_pursuit/nearest_segment"
TOPICS="$TOPICS /zed/zed_node/rgb/image_rect_color/compressed /pose_to_traj_error"
# Race to the moon topics
TOPICS="$TOPICS /race/left_lane /race/right_lane /race/mid_lane /race/trajectory"
# TOPICS="$TOPICS /zed/zed_node/rgb/image_rect_color"
# Heist topics
TOPICS="$TOPICS /debug_image /shell_points /detected_point /heist_state /detected_banana /traffic_light_point"

echo "Recording bag (run this on the car, edit topics in script)..."

# Start recording on the robot (Ctrl‑C to stop)
BAG=bag_$(date +%Y%m%d_%H%M%S)
ros2 bag record -o $BAG $TOPICS

# Print the bag file name
echo "Bag saved as $BAG"
echo "Use something like ***scp -i ~/.ssh/racecar_key -r \"racecar@192.168.1.107:~/racecar_ws/$BAG\" .*** to download it."
