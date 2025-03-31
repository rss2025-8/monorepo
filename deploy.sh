# Deploy code to the racecar with our SSH key, excluding Git files
rsync -avzhP -e "ssh -i ~/.ssh/racecar_key" --filter=':- .gitignore' --exclude='.git' . racecar@192.168.1.107:~/racecar_ws
