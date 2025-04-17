#!/bin/bash

echo "Deploying, building, and running code..."

./deploy.sh
ssh -i ~/.ssh/racecar_key racecar@192.168.1.107 "cd racecar_ws && ./colcon.sh && ./test.sh"
