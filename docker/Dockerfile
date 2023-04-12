
FROM ubuntu:20.04

# Minimal Docker container environment for TIDL-tools. Dramatically smaller than their official
# container (omitted some libraries) but includes everything needed for osrt compilation.

# WARNING: This container MUST be run with the --shm-size Docker flag set to some larger value. --shm-size=2gb seems to work.
# https://e2e.ti.com/support/processors-group/processors/f/processors-forum/1160791/tda4vm-bus-error-when-running-forward-pass-through-model-with-tidl_tools-on-pc

# On Ubuntu 18.04, I have found the following packages to also be necessary:
# sudo apt install cmake build-essential protobuf-compiler libprotobuf-dev libjpeg-dev python3.6-dev

# Note: Skipped ArmNN, TFLite, ARM gcc, and CPP deps

ENV DEVICE="j7"


RUN apt-get update && apt-get install -y \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y \
    python3.6 \
    python3-pip \
    python3.6-distutils \
    git \
    wget \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

ADD https://raw.githubusercontent.com/TexasInstruments/edgeai-tidl-tools/3dc359873f0b80c1e1a0b4eec5d1d02e35d4e532/requirements_pc.txt /
RUN python3.6 -m pip install -r /requirements_pc.txt

# Make sure to choose the version of the tools corresponding to the version of TIDL on your device.
RUN cd / \
    && wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_01-rc1/tidl_tools.tar.gz \
    # For TIDL 8.4:
    # && wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/tidl_tools.tar.gz \
    && tar -xzf tidl_tools.tar.gz \
    && rm tidl_tools.tar.gz

ENV TIDL_TOOLS_PATH="/tidl_tools/"
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TIDL_TOOLS_PATH:$TIDL_TOOLS_PATH/osrt_deps
