FROM nvcr.io/nvidia/isaac-lab:2.3.1

ARG WS=/workspace/r2dreamer

ENV ACCEPT_EULA=Y \
    LIVESTREAM=1

# Install required python packages
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install --upgrade pip
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install ruamel.yaml==0.19.1 torchrl==0.11.1 moviepy==1.0.3 tensorboard==2.17.1
# The base image already includes moviepy and tensorboard, but for compatibility
# reasons with the current logger implementation, we downgrade them here.

WORKDIR $WS

# Override entrypoint
ENTRYPOINT []
CMD ["/bin/bash"]
