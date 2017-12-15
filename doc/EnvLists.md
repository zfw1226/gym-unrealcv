#Env list of Gym-UnrealCV

we provide a set of pre-defined gym environments for various tasks.
The action spaces are various from discrete to continuous and the observation spaces are various from depth image to RGB-D image.

The details about these environments are shown in [the register file](../gym_unrealcv/__init__.py). We summarize the environments as below:

## Object Searching and Obstacle Avoidance
Task: find target object and avoid obstacle simultaneously.

UnrealEnv: [RealisticRendering](https://s3-us-west-1.amazonaws.com/unreal-rl/RealisticRendering_RL_3.10.zip)

Naming rule: `{Task}-{Scene}{Target}{ActionSpace}-{Version}`

- Search-RrDoorDiscrete-v0
- Search-RrDoorContinuous-v0
- Search-RrPlantsDiscrete-v0
- Search-RrPlantsContinuous-v0
- Search-RrSocketsDiscrete-v0
- Search-RrSocketsContinuous-v0

## Active Object Tracking
Task: actively track the target object.

UnrealEnv: [City1](https://s3-us-west-1.amazonaws.com/unreal-rl/SplineCharacterF.zip), [City2](https://s3-us-west-1.amazonaws.com/unreal-rl/SplineCharacterA.zip)

Naming rule: `{Task}-{Scene}{Target}{PathID}{AugmentEnv}-{Versrion}`

- Tracking-City1StefaniPath1Random-v0 
- Tracking-City1StefaniPath1Static-v0 
- Tracking-City1MalcomPath1Static-v0 
- Tracking-City1StefaniPath2Static-v0 
- Tracking-City2MalcomPath2Static-v0

Active Object Tracking Environment is used in 
`Luo, Wenhan, et al. "End-to-end Active Object Tracking via Reinforcement Learning."` [arXiv](https://arxiv.org/abs/1705.10561).
More details about the environment definition can be found in this paper.



