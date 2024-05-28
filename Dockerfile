ARG CUDA_VERSION=11.8.0
ARG OS_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}
ARG CUDA_VERSION
ARG OS_VERSION

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install pytorch and submodules
# echo "${CUDA_VERSION}" | sed 's/.$//' | tr -d '.' -- CUDA_VERSION -> delete last digit -> delete all '.'
RUN CUDA_VER=$(echo "${CUDA_VERSION}" | sed 's/.$//' | tr -d '.') && python3.10 -m pip install --no-cache-dir \
    torch==2.1.2+cu${CUDA_VER} \
    torchvision==0.16.2+cu${CUDA_VER} \
    --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VER}

RUN pip install \
    rich \
    PyYAML \
    Pillow \
    click \
    scipy

WORKDIR /workspace

CMD /bin/bash -l
