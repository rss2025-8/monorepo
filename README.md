# Team 8 Repo

All our code will be in here!

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
