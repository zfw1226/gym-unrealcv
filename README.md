Integrate UnrealCV with Openai Gym for Reinforcement Learning(RL)
===
In this tutorial, we show how to get started with installing environment, adding new envirnnment for specific RL tasks and train a DQN model for visual navigation in a realistic room.

![search1](https://i.imgur.com/esXQ0tI.gif)
![search2](https://i.imgur.com/fPVfRVt.gif)

**This branch is in experiment!!!**

Installation
===
## Dependencies
- Docker
- Nvidia-Docker
- UnrealCV
- Gym
- CV2
- Matplotlib
- Numpy
 
We recommend you to use [anaconda](https://www.continuum.io/downloads) to install and manage your python environment.

## Gym-UnrealCV

Install gym-unrealcv
```
git clone https://github.com/zfw1226/gym-unreal.git
cd gym-unrealcv
pip install -e . 
```
## Prepare Unreal Environment
You need prepare an unreal environment to run the demo as below. You can do it by running the script [RealisticRendering.sh](RealisticRendering.sh)
```buildoutcfg
sh RealisiticRendering.sh
```
To run environments based on ArchinteriorsVol2Scene1, you need run script [Arch1.sh](Arch1.sh) to get the ArchinteriorsVol2Scene1 binary
```buildoutcfg
sh Arch1.sh
```

There are two ways to run the unreal environment in gym-unrealcv. One need install [docker](https://docs.docker.com/engine/installation/linux/ubuntu/#install-from-a-package) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker), another do not need install anything else.
Currently the docker-free version only support running an unreal environment in a computer for the confliction of server IP.
If you need running multiple unreal environments in a computer in parallel, [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and [docker](https://docs.docker.com/engine/installation/linux/ubuntu/#install-from-a-package) are required.

**To make gym-unrealcv easy to run, the default config do not need install ``Docker``.**

### Run without Docker
There is nothing required to be installed.

### Run with Docker
For the reason that ```nvidia-docker``` supports ```Linux```  and ```Nvidia GPU```only , you will have to install and run our openai-gym environment in ```Linux``` system with ```Nvidida GPU```.
As the unreal environment with UnrealCV runs inside Docker containers, you are supposed to install [docker](https://docs.docker.com/engine/installation/linux/ubuntu/#install-from-a-package) first. If you use Linux, you can run scripts as below:
```
curl -sSL http://acs-public-mirror.oss-cn-hangzhou.aliyuncs.com/docker-engine/internet | sh -
```
Once docker is installed sucessfully, you are able to run ```docker ps``` and get something like this:
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

Run a simple envirnment
===
Once ```gym-unrealcv``` is installed sucessfully, you will see that your agent walking randomly in first-person view, after you run:
```
cd example/random
python random_agent.py
```
It will take a few minutes for the image to pull if you runs environment based on docker at the first time. 

After that, if all goes well，a predefined gym environment ```Search-RrPlantDiscrete-v0``` will be launched.
And then you will see that your agent is moving around the realistic room randomly.


Run a reinforcement learning example
===
Besides, we provide an example to train an agent to visual navigation by searching for specific object and avoiding obstacle simultaneously in ``Search-RrPlantDiscrete-v0`` environement using [Deep Q-Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).
### Dependences
To run this example, you should make sure that you have installed all the dependences. We recommend you to use [anaconda](https://www.continuum.io/downloads) to install and manage your python environment.
- Keras(Tested with v1.2)
- Theano or thensorflow
- Openai gym(>=v0.7)
- cv2
- matplotlib
- numpy

To use Keras(v1.2), you should run
```
pip install keras==1.2
```

You can start the training process with default parameters by runinng the following script:
```
cd example/dqn
python run.py
```
You will see a window like this:

![show](https://i.imgur.com/HyOVKD4.png)

While the ```Collision``` button turning red, a collision is detected.
While the ```Trigger``` button turning red, the agent is taking an aciton to ask the environment if it is seeing the target in a right place. 
You can change some parameteters in [```example/dqn/constant.py```]()
You can change the architure of DQN in [```example/dqn/dqn.py```]() 

Visualization
===
You can display a graph showing the history episode rewards by running the following script:
```
cd example/utility
python reward.py 
```
![reward](https://i.imgur.com/W039bbs.jpg)


You can display a graph showing the trajectory by running the following script:
```
cd example/utility
python trajectory.py
```
![trajectory](https://i.imgur.com/PKpKHNR.png)

- The ```green points``` represent where the agents realized that they had found a good view to observe the target object and got positive reward from  the environment.At the same time, the episode is finished. 
- The ```purple points``` represnet where collision detected collision, agents got negative reward. At the same time, the episode terminated. 
- The ```red lines```  represent the trajectories that the agents found taget object sucessfully in the end.
- The ```black lines``` represent the trajectories of agents that did not find the target object in the end.



