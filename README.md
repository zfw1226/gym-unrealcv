Integrate Unreal Engine with OpenAI Gym for Reinforcement Learning based on UnrealCV
===
# Introduction
**This project integrates Unreal Engine with OpenAI Gym for visual reinforcement learning based on [UnrealCV](http://unrealcv.org/).**
In this project, you can run your RL algorithms in various realistic UE4 environments easily without any knowledge of Unreal Engine and UnrealCV.
The framework of this project is shown as below:

![framework](./doc/framework.JPG)

- ```UnrealCV``` is the basic bridge between ```Unreal Engine``` and ```OpenAI Gym```.
- ```OpenAI Gym``` is a toolkit for developing RL algorithm, compatible with any numerical computation library, such as Tensorflow or Theano. 
 
The tutorial will show you how to get started with **installing environment, running an agent in an environment, adding a new environment and training a reinforcement learning agent for visual navigation.**

<div align="center">

![search1](./doc/search1.gif)
![search2](./doc/search2.gif)

Snapshots of RL based visual navigation for object searching and obstacle avoidance.

</div>


# Install Environment
## Dependencies
- Docker
- Nvidia-Docker
- UnrealCV
- Gym
- CV2
- Matplotlib
- Numpy
 
We recommend you to use [anaconda](https://www.continuum.io/downloads) to install and manage your python environment.
Considering performance, we use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run the unreal environment. 
For the reason that ```nvidia-docker``` supports ```Linux```  and ```Nvidia GPU```only , 
you will have to install and run our openai-gym environment in ```Linux``` system with ```Nvidida GPU```.
```CV2``` is used for images processing, like extracting object mask and bounding box.```Matplotlib``` is used for visualization.


## Docker
As the unreal environment with UnrealCV runs inside Docker containers, you are supposed to install [docker](https://docs.docker.com/engine/installation/linux/ubuntu/#install-from-a-package) first. If you use Linux, you can run scripts as below:
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


## Gym-UnrealCV

It is easy to install gym-unrealcv, just run
```
git clone https://github.com/zfw1226/gym-unrealcv.git
cd gym-unrealcv
pip install -e . 
```
While installing gym-unrealcv, dependencies including [OpenAI Gym](https://github.com/openai/gym), [docker-py](https://github.com/docker/docker-py), cv2 and matplotlib are installed.


## Prepare Unreal Environment
You need prepare an unreal environment to run the demo as below. You can do it by running the script [RealisiticRendering.sh](RealisticRendering.sh)
```
sh RealisiticRendering.sh
```

# Usage

## Run a random agent in an unreal environment


Once ```gym-unrealcv``` is installed successfully, you can test it by running:
```
cd example/random
python random_agent.py
```
It will take a few minutes for the image to pull the first time. After that, if all goes well，a simple predefined gym environment ```Unrealcv-Simple-v0``` wiil be launched.And then you will see that an agent is moving around the realistic room randomly in first-person view.

## Add a new unreal environment

More details about adding new unreal environment for your Rl tasks is [here](./doc/addEnv.md).

## Training a reinforcement learning agent

Besides, we provide an example to train an agent for visual navigation by searching for specific object and avoiding obstacle simultaneously in [Unrealcv-Search-v0](https://github.com/zfw1226/gym-unrealcv/blob/master/gym_unrealcv/envs/unrealcv_search.py) environement using [Deep Q-Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).

### Dependencies
To run this example, some additional dependencies should be installed for deep reinforcement learning. 
- [Keras](https://keras.io/#switching-from-tensorflow-to-theano)(Tested with v1.2)
- Theano or thensorflow

To install Keras(v1.2), you should run
```
pip install keras==1.2
```
Please see [this instruction](https://keras.io/backend/) to switch backend between ```Theano``` and ```Tensorflow```

If you use  the ```Theano``` backend, please see [this instruction](http://deeplearning.net/software/theano/library/config.html) to config gpu.

If you use ```Tensorflow```backend, please set ```DEVICE_TF``` in [constants.py](./example/dqn/constants.py) to config gpu

### Training an agent
You can start the training process with default parameters by running the following script:
```
cd example/dqn
python run.py
```
The default target objects of [Unrealcv-Search-v0](./gym_unrealcv/envs/unrealcv_search.py) are two potted plant in this room. 
While the env reset, the agent restart from one of start positions in the list  ```self.origin``` with a random yaw angle.

You can change some parameteters in [constants.py](./example/dqn/constants.py)
if you set ```SHOW``` is ```True```, You will see a window like this to monitor the agent while training:

<div align="center">

![show](./doc/show.PNG)

</div>

- While the ```Collision``` button turning red, a collision is detected.
- While the ```Trigger``` button turning red, the agent is taking an action to ask the environment if it is seeing the target in a right place. 

if you set ```Map``` is ```True```, you will see a window showing the trajectory of the agent like this:

<div align="center">

![map](./doc/map.gif)

</div>

- The ```green points``` represent where the agents realized that they had found a good view to observe the target object and got positive reward from  the environment.At the same time, the episode is finished. 
- The ```purple points``` represent where collision detected collision, agents got negative reward. At the same time, the episode terminated. 
- The ```red points``` represent where the targets are.
- The ```blue points``` represent where the agent start in a new episode.
- The ```red lines```  represent the trajectories that the agents found taget object sucessfully in the end.
- The ```black lines``` represent the trajectories of agents that did not find the target object in the end.
- The ```blue line``` represents the trajectory of agent in the current episode.

You can change the architecture of DQN in [dqn.py](./example/dqn/dqn.py) 


## Visualization

You can display a graph showing the history episode rewards by running the following script:
```
cd example/utility
python reward.py 
```

<div align="center">

![reward](https://i.imgur.com/W039bbs.jpg)

</div>


You can display a graph showing the trajectory by running the following script:
```
cd example/utility
python trajectory.py
```

<div align="center">

![trajectory](https://i.imgur.com/PKpKHNR.png)

</div>

- The ```green points``` represent where the agents realized that they had found a good view to observe the target object and got positive reward from  the environment.At the same time, the episode is finished. 
- The ```purple points``` represent where collision detected collision, agents got negative reward. At the same time, the episode terminated. 
- The ```red lines```  represent the trajectories that the agents found taget object sucessfully in the end.
- The ```black lines``` represent the trajectories of agents that did not find the target object in the end.

##Contact
if you have any suggestion or interested in using Gym-UnrealCv, get in touch at [zfw1226@gmail.com](zfw1226@gmail.com).

## Cite
If you use Gym-UnrealCV in your academic research, we would be grateful if you could cite it as follow:
```buildoutcfg
@misc{gymunrealcv2017,
    author = {Fangwei Zhong, Weichao Qiu, Tingyun Yan, Alan Yuille, Yizhou Wang},
    title = {Gym-UnrealCV: Realistic virtual worlds for visual reinforcement learning},
    howpublished={Web Page},
    url = {https://github.com/unrealcv/gym-unrealcv},
    year = {2017}
}
```




