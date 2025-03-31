# Edit this to quickly launch whatever we're working on rn

echo "Building..."
PYTHONWARNINGS="ignore" colcon build --symlink-install
source install/setup.bash

echo "Launching..."
ros2 launch visual_servoing parking_deploy.launch.xml type:=line
