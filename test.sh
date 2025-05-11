# Edit this to launch whatever we're working on rn

source install/setup.bash
ros2 launch race_to_the_moon visual_test.launch.xml bag_file:=local/bags/racetrack/
# ros2 launch race_to_the_moon visual_test.launch.xml bag_file:=local/bags/actualslow/
ros2 launch race_to_the_moon act_race_to_the_moon.launch.xml