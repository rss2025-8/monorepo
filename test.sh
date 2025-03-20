# Edit this to quickly launch whatever lab we're working on rn

echo "Building..."
colcon build --symlink-install
echo "Launching..."
source install/setup.bash
ros2 launch visual_servoing parking_deploy.launch.xml
