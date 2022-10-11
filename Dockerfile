FROM tensorflow/tensorflow:2.7.1-gpu as tf_base

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
# Minimal required libs 
RUN apt-get update \
    && apt-get upgrade -y \ 
    #Install basic utilities.
    && apt install -qy libglib2.0-0 openssh-server ffmpeg libsm6 libxext6 \
    && apt-get install -y --no-install-recommends git wget curl vim gcc g++ cmake unzip bzip2 build-essential ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -r /tmp/requirements.txt 

