# Edit this to quickly launch whatever we're working on rn

echo "Building..."
PYTHONWARNINGS="ignore" colcon build --symlink-install
source install/setup.bash

echo "Launching (Lab 6 on car)..."
ros2 launch path_planning real.launch.xml
