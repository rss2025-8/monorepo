# Builds and sources without extra warnings
PYTHONWARNINGS="ignore" colcon build --symlink-install
source install/setup.bash
