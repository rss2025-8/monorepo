#!/bin/bash

echo "Deploying and running code (no rebuild)..."

./deploy.sh
ssh -i ~/.ssh/racecar_key racecar@192.168.1.107 "cd racecar_ws && ./test.sh"
