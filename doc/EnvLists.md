#Env list of Gym-UnrealCV
Based on two virtual worlds([RealisticRendering](https://s3-us-west-1.amazonaws.com/unreal-rl/RealisticRendering_RL_3.10.zip) and [ArchinteriorsVol2Scene1](http://cs.jhu.edu/~qiuwch/release/unrealcv/ArchinteriorsVol2Scene1-Linux-0.3.10.zip)),
we provide a set of pre-defined gym environments for task that learn to object searching and obstacle avoidance simultaneously.
The action spaces are various from discrete to continuous and the observation spaces are various from depth image to RGB-D image.

The details about these environments are shown in [the register file](../gym_unrealcv/__init__.py). We summarize the environments as below:
##Training set
- **Search-RrDoorDiscrete-v0**
- **Search-RrDoorContinuous-v0**
- **Search-RrMultiPlantsDiscrete-v0**
- **Search-RrMultiPlantsContinuous-v0**
- **Search-RrMultiSocketsDiscrete-v0**
- **Search-RrMultiSocketsContinuous-v0**
- **Search-Arch1DoorDiscrete-v0**
- **Search-Arch1DoorContinuous-v0**

##Testing set
- **Search-RrDoorDiscreteTest-v0**
- **Search-RrDoorContinuousTest-v0**
- **Search-RrMultiPlantsDiscreteTest-v0**
- **Search-RrMultiPlantsContinuousTest-v0**
- **Search-RrMultiSocketsDiscreteTest-v0**
- **Search-RrMultiSocketsContinuousTest-v0**
- **Search-Arch1DoorDiscreteTe'jy.n`hwj.nhj.nst-v0**
- **Search-Arch1DoorContinuousTest-v0**

The difference between the ```training set``` and the ```testing set``` is
the way to generate the starting position.
- In the training set, the agent will reset from a waypoint which is generated and selected by the waypoint module according to historical trajectory.
- In the testing set, the agent will reset from a set of pre-recorded test points randomly.