# Running Gym-UnrealCV with Docker
For the reason that ```nvidia-docker``` supports ```Linux```  and ```Nvidia GPU```only , you will have to install and run our gym-unrealcv environment in ```Linux``` system with ```Nvidida GPU```.
As the unreal environment with UnrealCV runs inside Docker containers, you are supposed to install [docker](https://docs.docker.com/engine/installation/linux/ubuntu/#install-from-a-package) firstly. If you use Linux, you can run scripts as below:
```
curl -sSL http://acs-public-mirror.oss-cn-hangzhou.aliyuncs.com/docker-engine/internet | sh -
```
Once docker is installed successfully, you are able to run ```docker ps``` and get something like this:
```
$ docker ps
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
```

To speed up the frame rate of the environment, you need install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/wiki) to utilize NVIDIA GPU in docker.
```
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
```
Test nvidia-docker
```
nvidia-docker run --rm nvidia/cuda nvidia-smi
```
You should be able to get the same result as you run ```nvidia-smi``` in your host.

And Then, [docker-py](https://github.com/docker/docker-py) a python library for the Docker Engine API is required to manage docker in python.
```buildoutcfg
pip install docker>= 2.2
```

**To use ``nvidia-docker``, you need modify variable ``use_docker`` to ``True``in [gym_unrealcv/__init__.py](../gym_unrealcv/__init__.py)**
