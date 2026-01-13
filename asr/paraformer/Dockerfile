# Get pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

LABEL name="zerolan-core-asr-paraformer" \
    maintainer="AkagawaTsurunaki <AkagawaTsurunaki@outlook.com>" \
    lastupdate="2024-11-05"

# Create workspace folder and set it as working directory
RUN mkdir -p /workspace/services/ && \
    mkdir -p /workspace/models/
WORKDIR /workspace

# Set the timezone
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Install git
RUN apt-get install git -y && apt-get install git-lfs
RUN apt-get install wget

# Download model
RUN cd /workspace/models/ && \
    git lfs install && \
    git clone https://www.modelscope.cn/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1.git

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Update PATH environment variable
ENV PATH="/workspace/miniconda3/bin:${PATH}"

# Initialize conda
RUN conda init bash

# Create and activate conda environment
# 接受 Conda 服务条款并创建 conda 环境
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n paraformer python=3.10 -y && \
    echo "source activate paraformer" > ~/.bashrc

# 修复 ENV 格式警告（key=value 格式）
ENV PATH=/workspace/miniconda3/envs/paraformer/bin:$PATH

# Clone the repository
RUN cd /workspace/services/ && \
    git clone https://github.com/AkagawaTsurunaki/zerolan-core.git
    
# [Dev] Check to branch dev
RUN cd /workspace/services/zerolan-core && \ 
    git checkout dev

# Install required dependencies
# NOTE: Users who are not in China please comment these mirror config commands
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple && \
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# https://mirrors.aliyun.com/pytorch-wheels https://mirrors.aliyun.com/pytorch-wheels/cu121/
# 安装指定版本的 torch 和匹配的 torchaudio，优先使用阿里云 PyTorch 镜像
# 彻底切换 pip 源为阿里云（替代不可用的清华源）
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.aliyun.com

# 一次性安装基础依赖 + 指定版本的 torch 和 torchaudio（优先阿里云 PyTorch 镜像）
RUN pip install filelock typing-extensions sympy networkx jinja2 fsspec requests && \
    pip install torch==2.3.1 torchaudio==2.3.1 -f https://mirrors.aliyun.com/pytorch-wheels/cu121/

# [Dev] Install zerolan-data
# 优化 Git 克隆参数，解决 TLS 连接中断问题
RUN git config --global http.postBuffer 524288000 && \
    git config --global http.sslVerify false && \
    git config --global http.lowSpeedLimit 0 && \
    git config --global http.lowSpeedTime 999999 && \
    pip install --default-timeout=1000 git+https://github.com/AkagawaTsurunaki/zerolan-data.git@dev
    
RUN cd /workspace/services/zerolan-core && \
    pip install -r requirements.txt && \
    pip install -r ./asr/paraformer/requirements.txt

# Install ffmpeg
RUN apt-get install ffmpeg -y

# Copy config file
RUN cd /workspace/services/zerolan-core && \
    cp config.template.yaml config.yaml