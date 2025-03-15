#https://docs.docker.com/build/building/packaging/

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y python3 python3-pip
RUN DEBIAN_FRONTEND=noninteractive apt-get install keyboard-configuration
RUN apt install nvidia-driver-525 -y
RUN apt-get install vim  -y
RUN apt-get install gdal-bin -y
RUN apt-get install libgdal-dev -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt install libcairo2-dev pkg-config python3-dev -y
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN export C_INCLUDE_PATH=/usr/include/gdal


COPY repo / src/repo/
COPY pyproject.toml /src 

WORKDIR /src
RUN pip install . 
RUN pip install torch==1.12.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113