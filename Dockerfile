FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NUMBA_CACHE_DIR=/tmp \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Install necessary packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create an ICD config file to register the NVIDIA EGL implementation with glvnd.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Set the working directory
WORKDIR /workspace

# Install requirements
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Set the default MuJoCo rendering backend to EGL, which is compatible with headless
# environments and does not require a display server.
ENV MUJOCO_GL=egl
