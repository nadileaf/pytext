# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

FROM ubuntu:16.04

# Install Caffe2 + dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  libgoogle-glog-dev \
  libgtest-dev \
  libiomp-dev \
  libleveldb-dev \
  liblmdb-dev \
  libopencv-dev \
  libopenmpi-dev \
  libsnappy-dev \
  openmpi-bin \
  openmpi-doc \
  python-dev \
  python-pip
RUN pip install --upgrade pip
RUN pip install setuptools wheel
RUN pip install future numpy protobuf typing hypothesis pyyaml
RUN apt-get install -y --no-install-recommends \
      libgflags-dev \
      cmake
RUN git clone https://github.com/pytorch/pytorch.git
WORKDIR pytorch
RUN git submodule update --init --recursive
RUN python setup.py install

# Install libcurl
RUN apt-get install -y libcurl4-openssl-dev

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy local files to /app
COPY . /app
WORKDIR /app

# Open ports
EXPOSE 9090

# Run app.py when the container launches
CMD ["python", "server.py"]
