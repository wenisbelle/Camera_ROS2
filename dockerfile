FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Minimal setup
RUN apt-get update \
 && apt-get install -y locales lsb-release
ARG DEBIAN_FRONTEND=noninteractive
RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
SHELL [ "/bin/bash" , "-c" ]

# Install ROS2 Humble
RUN apt update \
 && apt install -y --no-install-recommends curl \
 && apt install -y --no-install-recommends gnupg2 \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y net-tools gedit
 
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt update \
 && apt install -y ros-humble-desktop

RUN apt-get install -y python3-rosdep \
 && apt-get install -y python3-colcon-common-extensions

RUN apt-get install -y git

RUN apt-get update

RUN apt-get install -y python3-pip \
    python3-pip \
    python3-dev \
    python3-opencv \
    ros-humble-cv-bridge

RUN pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
RUN pip3 install tensorflow

RUN pip3 install "numpy<2"
RUN pip3 install "pybind11>=2.12"

RUN python3 -c "import torch; print(torch.cuda.is_available())" && \
    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

RUN apt-get install -y ros-humble-librealsense2* \
                        ros-humble-realsense2-* 

RUN apt-get install -y python3-opencv

RUN apt-get install -y libopencv-dev

# Dependencies required by MediaPipe
RUN apt-get update && apt-get install -y \
    ffmpeg \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0

# MediaPipe install
RUN pip3 install mediapipe

RUN source /opt/ros/humble/setup.bash

RUN mkdir -p /Intel_Camera/src

WORKDIR /Intel_Camera

RUN colcon build

RUN echo "export ROS_DOMAIN_ID=30" >> ~/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc && \
    #echo "export CYCLONEDDS_URI=/turtlebot_ws/src/dds_config/cyclonedds_unicast_wlan.xml" >> ~/.bashrc && \
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /Intel_Camera/install/setup.bash" >> ~/.bashrc
 
    


