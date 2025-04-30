# Team 8 Repo

All our code will be in here!

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
