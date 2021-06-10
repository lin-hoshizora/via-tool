#! /bin/bash
if [ ! -n "$(sudo docker ps | grep model)" ]
then
echo "run model_server..."
sudo docker run -d --restart=always -v /home/almexdi/Smapa-Terminal-Backend3/config/:/app/config --network=host model_server:2.1
else
echo "model_server runing..."
fi

if [ ! -n "$(sudo docker ps | grep via)" ]
then
echo "run via-tool..."
sudo docker run -d --restart=always -v ~/ocr-debug-jupyter-app/via-tool-json-maker:/app --network=host  via-tool:1.0 bash -c "python3 -m http.server 8777"
else
echo "via-tool runing..."
fi

echo "press any key..."
read

