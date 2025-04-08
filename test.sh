# Edit this to quickly launch whatever we're working on rn

echo "Building..."
PYTHONWARNINGS="ignore" colcon build --symlink-install
source install/setup.bash

echo "Launching (**remember to also re-launch map!**)..."
ros2 launch localization localize.launch.xml env:=act
