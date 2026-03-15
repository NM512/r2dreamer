# Docker

If you prefer not to install dependencies locally, or if you want to train your models on a containerized remote machine, you can use the provided Dockerfile to build an image with all dependencies pre-installed.

The only prerequisites are [Docker](https://docs.docker.com/get-docker/) and, on your deployment machine, the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU support.

To build the Docker image, run the following command from the root of the repository:

```bash
docker build -f Dockerfile -t r2dreamer:local .
```
You can replace the `-t` argument with any image name you like. The command above will build and tag the image as `r2dreamer:local`.

Then start a container from the built image with:

```bash
docker run -it -d --rm \
    --gpus=all \
    --network=host \
    --volume=$PWD:/workspace \
    --name=r2dreamer-container \
    r2dreamer:local
```

You can then connect to the running container and execute your training scripts. For example, to run R2-Dreamer on DMC Walker Walk:

```bash
# Connect to the running container
docker exec -it r2dreamer-container bash

# And then inside the container:
python3 train.py env=dmc_vision env.task=dmc_walker_walk

# Alternatively, combine it with the docker exec command and use the -d flag to run in detached mode:
docker exec -it -d r2dreamer-container bash -c "python3 train.py env=dmc_vision env.task=dmc_walker_walk"
```

To monitor training progress with TensorBoard, run the following command in a separate terminal on your host machine:

```bash
docker exec -it r2dreamer-container tensorboard --logdir ./logdir
```

The TensorBoard dashboard will then be available at `http://localhost:6006/`.

## IsaacLab

The IsaacLab image is built on top of `nvcr.io/nvidia/isaac-lab` and includes Isaac Sim alongside all r2dreamer dependencies. Because Isaac Sim is a large download, the build takes considerably longer and produces a much larger image than the base r2dreamer image.

> **Note:** The IsaacLab image requires an NVIDIA GPU with drivers compatible with Isaac Sim. Refer to the [Isaac Sim Requirements](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/installation/requirements.html) for the minimum driver version.

Build the image from the root of the repository:

```bash
docker build -f IsaacLab.Dockerfile -t r2dreamer-isaaclab:local .
```

Then start a container from the built image with:

```bash
docker run -it -d --rm \
    --gpus=all \
    --network=host \
    --volume=$PWD:/workspace/r2dreamer \
    --name=r2dreamer-isaaclab-container \
    r2dreamer-isaaclab:local
```

Connect to the running container and execute your training scripts:

```bash
# Connect to the running container
docker exec -it r2dreamer-isaaclab-container bash

# And then inside the container:
python ./recipes/isaaclab/cartpole/train_cartpole.py env=isaaclab_vision env.task=isaaclab_cartpole_balance_dmc
```

See [isaaclab.md](isaaclab.md) for the full list of cartpole variants and other task options.

### Visualizing with WebRTC

The official solution for connecting to a remote or containerized Isaac Sim instance is the [NVIDIA Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/manual_livestream_clients.html). The IsaacLab image ships with `LIVESTREAM=1` baked in, so the WebRTC streaming server starts automatically alongside the simulation, no extra configuration is needed.

If you run the container on your local machine, you can connect the streaming client to `127.0.0.1`. If you want to connect to a container running on a remote machine, you have to export the environment variable `PUBLIC_IP=x.x.x.x`, where `x.x.x.x` is your remote machine's IP address, inside the container first.
