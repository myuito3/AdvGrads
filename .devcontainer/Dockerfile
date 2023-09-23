# Define base image.
FROM tensorflow/tensorflow:1.15.0-gpu-py3

# Downgrade the public key.
RUN rm -f /etc/apt/sources.list.d/cuda.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && rm -f cuda-keyring_1.0-1_all.deb
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git

# Bash as default entrypoint.
CMD /bin/bash -l