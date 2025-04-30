# Edit this to launch whatever we're working on rn

LAUNCH_CMD="ros2 launch path_planning real.launch.xml"

source install/setup.bash
echo "Launch command: $LAUNCH_CMD"
# Ensure there is no memory leak here (node names should not have high numbers)
exec $LAUNCH_CMD
