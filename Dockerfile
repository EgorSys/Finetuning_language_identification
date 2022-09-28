FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get autoremove && apt-get autoclean
RUN apt-get install -y -q apt-utils python3-pip
RUN apt-get install -y apt-utils ffmpeg python3-pip portaudio19-dev wget
RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip3 --default-timeout=1500 install -r requirements.txt

ARG PROJECT=finetuning
ARG PROJECT_DIR=/${PROJECT}
RUN mkdir -p $PROJECT_DIR
WORKDIR $PROJECT_DIR
