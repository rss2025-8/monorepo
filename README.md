# Team 8 Repo

All our code will be in here!

## Final Challenge Part A: Shrink Ray Heist

Testing in sim:
```sh
ros2 launch shrinkray_heist sim.launch.xml
```

Testing on car:
```sh
# Consider changing power modes
sudo nvpmodel -m 2
sudo nvpmodel -q –verbose
sudo reboot
# Wait a bit...
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed
ros2 launch shrinkray_heist real.launch.xml
```

Recording a bag:
```sh
# Run on the car, modify recorded topics in record_bag.sh locally before deploying
./record_bag.sh
```

## Final Challenge Part B: Race to the Moon

Visualizing/testing a bag in simulation:
```sh
# Replace bag_file with the path to your bag file
ros2 launch race_to_the_moon visual_test.launch.xml bag_file:=local/bags/racetrack/
ros2 launch race_to_the_moon bag_visual_test.launch.xml bag_file:=local/bags/actual_1mps
ros2 launch race_to_the_moon bag_visual_test.launch.xml bag_file:=local/bags/actual_2.5mps
```

On the racecar:
```sh
# Consider changing power modes
sudo nvpmodel -m 2
sudo nvpmodel -q –verbose
sudo reboot
# Wait a bit...
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed
ros2 launch race_to_the_moon act_race_to_the_moon.launch.xml
```

Recording a bag:
```sh
# Run on the car, modify recorded topics in record_bag.sh locally before deploying
./record_bag.sh
```

## Lab 6: Path Planning

Testing path planning/following in simulation:
```sh
# Load the planning.rviz file in path_planning for a cool visual!
ros2 launch path_planning sim_plan_follow.launch.xml  # Ground truth odometry
ros2 launch path_planning pf_sim_plan_follow.launch.xml  # With MCL
ros2 launch path_planning noisy_sim_plan_follow.launch.xml  # More realistic Ackermann dynamics
```

Testing path planning/following on the car:
```sh
# Load the planning.rviz file in path_planning for a cool visual!
ros2 launch path_planning real.launch.xml
```

Use "2D Goal Pose" to set end point, "2D Pose Estimate" to set start point.

## Lab 5: Monte Carlo Localization

Testing localization in simulation:
```sh
ros2 launch wall_follower wall_follower.launch.xml
ros2 launch localization localize.launch.xml env:=sim
# Launch this last, then make sure the debug logs for the localizer have no warnings!
ros2 launch racecar_simulator simulate.launch.xml
```

Testing localization on the car:
```sh
ros2 launch localization localize.launch.xml env:=act
# Launch this last, then make sure the debug logs for the localizer have no warnings!
ros2 launch racecar_simulator localization_simulate.launch.xml
```

## Lab 4: Vision

To run vision tests:
```sh
python3 cv_test.py [citgo/cone/map]
```

To run the parking controller in simulation:
```sh
ros2 launch racecar_simulator simulate.launch.xml
ros2 launch visual_servoing parking_sim.launch.xml
```

To run the parking controller on the actual racecar (turn the camera on first):
```sh
ros2 launch visual_servoing parking_deploy.launch.xml type:=parking
```

To run the line follower on the actual racecar (turn the camera on first):
```sh
ros2 launch visual_servoing parking_deploy.launch.xml type:=line
```

## Lab 3: Wall-Following on the Racecar

To run the test code in simulation:
```sh
ros2 launch wall_follower launch_test_sim.launch.py
ros2 launch wall_follower launch_test.launch.py
```

To run the wall follower + safety controller in simulation:
```sh
ros2 launch wall_follower launch_test_sim.launch.py
ros2 launch wall_follower racecar.launch.xml env:=sim
```

To run the wall follower + safety controller on the physical racecar:
```sh
ros2 launch wall_follower racecar.launch.xml env:=act
```
