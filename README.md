# Team 8 repo

All our code will be in here!

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
