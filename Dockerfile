FROM ubuntu:18.04

ARG project_dir=/app/
ADD requirements.txt $project_dir
WORKDIR $project_dir


RUN apt-get update \
&& apt-get install -y language-pack-ja-base language-pack-ja \
&& echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc

RUN apt update && apt install -y python3-pip

RUN pip3 install --upgrade pip

RUN python3 -m pip install -r requirements.txt


