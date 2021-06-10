#!/bin/bash
sudo docker exec -it $(sudo docker ps | grep via | awk '{print $1}') bash -c "export LANG=ja_JP.UTF-8;cd /app/;python3 main.py"
