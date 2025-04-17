# Edit this to launch whatever we're working on rn

LAUNCH_CMD="ros2 launch path_planning real.launch.xml"

source install/setup.bash
echo "Launch command: $LAUNCH_CMD"
eval $LAUNCH_CMD
