FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    wget && \
    rm -rf /var/lib/apt/lists/*

RUN pip install \
    rich \
    PyYAML \
    Pillow \
    click \
    scipy

WORKDIR /workspace

CMD /bin/bash -l
