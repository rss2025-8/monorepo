#!/usr/bin/env bash

TOPICS="/map /robot_description /tf /tf_static /pf/pose/odom"  # Core topics
TOPICS="$TOPICS /followed_trajectory/start_point /followed_trajectory/end_pose /followed_trajectory/path /planned_trajectory/start_point /planned_trajectory/end_pose /planned_trajectory/path"  # Path planner
TOPICS="$TOPICS /pure_pursuit/drive_line /pure_pursuit/driving_arc /pure_pursuit/lookahead_circle /pure_pursuit/lookahead_point /pure_pursuit/nearest_segment"  # Trajectory follower
echo "Recording bag on the car (run this locally, edit topics in script)..."

# Start recording on the robot (Ctrl‑C to stop)
KEY=~/.ssh/racecar_key
HOST=racecar@192.168.1.107
BAG=bag_$(date +%Y%m%d_%H%M%S)
ssh -i "$KEY" "$HOST" "ros2 bag record -o $BAG $TOPICS"

# Pull the bag back and clean up
scp -i "$KEY" -r "$HOST:$BAG" .
ssh -i "$KEY" "$HOST" "rm -rf $BAG"

# Print the bag file name
echo "Bag saved as $BAG"
