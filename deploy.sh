rsync -avzhP -e "ssh -i ~/.ssh/racecar_key" --filter=':- .gitignore' . racecar@192.168.1.107:~/racecar_ws
