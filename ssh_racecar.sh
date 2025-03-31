# SSH to the racecar with our key and port forwarding for RViz
ssh -i ~/.ssh/racecar_key -L 6081:localhost:6081 racecar@192.168.1.107
