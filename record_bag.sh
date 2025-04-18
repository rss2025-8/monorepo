#!/usr/bin/env bash

# Core topics
TOPICS="/map /robot_description /tf /tf_static /pf/pose/odom"
# Sensors
TOPICS="$TOPICS /scan /vesc/odom"
# Path planner visuals
TOPICS="$TOPICS /followed_trajectory/start_point /followed_trajectory/end_pose /followed_trajectory/path /planned_trajectory/start_point /planned_trajectory/end_pose /planned_trajectory/path"
# Pure pursuit visuals
TOPICS="$TOPICS /pure_pursuit/drive_line /pure_pursuit/driving_arc /pure_pursuit/lookahead_circle /pure_pursuit/lookahead_point /pure_pursuit/nearest_segment"
echo "Recording bag (run this on the car, edit topics in script)..."

# Start recording on the robot (Ctrl‑C to stop)
KEY=~/.ssh/racecar_key
HOST=racecar@192.168.1.107
BAG=bag_$(date +%Y%m%d_%H%M%S)

ros2 bag record -o $BAG $TOPICS

# Pull the bag back and clean up
# scp -i "$KEY" -r "$HOST:~/racecar_ws/$BAG" .
# ssh -i "$KEY" "$HOST" "rm -rf $BAG"
# scp -i ~/.ssh/racecar_key -r "racecar@192.168.1.107:~/racecar_ws/$BAG" .
# Print the bag file name
echo "Bag saved as $BAG"
echo "Use something like ***scp -i ~/.ssh/racecar_key -r \"racecar@192.168.1.107:~/racecar_ws/$BAG\" .*** to download it."
