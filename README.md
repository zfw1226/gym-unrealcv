## About
This project shows some RL(DQN,DDPG) demos for unrealcv(http://unrealcv.org/).
This project is under construction.

## Requirements

- Keras(Tested with v1.2 Theano Backend)
- Openai gym(>=v0.7)
- cv2
- matplotlib
- numpy

# Installation
Install openai gym
```
$ git clone https://github.com/openai/gym
$ cd gym
$ pip install -e .
```

Install the gym-unreal environment
```
$ git clone https://github.com/zfw1226/gym-unreal.git -b DQN
$ cd gym-unreal
$ pip install -e .
```
## How to run dqn
Download the RealisticRendering UE4 Environment from unrealcv(http://unrealcv.org/)
then modify the the path of the UE4 env in unrealcv_env.py
Then run the dqn training code
```
$ python examples/dqn/run.py
```
You can adjust hyper-parameters in examples/dqn/constants.py.

Show the Result
```
$ python examples/dqn/plot/reward.py
```
