# Environment List

we provide a set of pre-defined gym environments for various tasks, including **objects searching**, **active object tracking**, and **robotic arm control**.
The ``Action Spaces`` could be Discrete and Continuous, and the ``Observation Spaces`` could be Depth, Color, RGB-D image.

The details about these environments are shown in [the register file](../gym_unrealcv/__init__.py). We summarize the environments as below:

## Object Searching
Goal: find the target object and avoid collision.

Binaries: [RealisticRoom](https://www.cs.jhu.edu/~qiuwch/unrealcv/binaries/RealisticRendering_RL_3.10.zip), [Arch1](https://www.cs.jhu.edu/~qiuwch/release/unrealcv/ArchinteriorsVol2Scene1-Linux-0.3.10.zip)

Naming rule: `UnrealSearch-{Scene}{Target}-{ActionSpace}{ObsSpace}-{Version}`
- `{Scene}`: RealisticRoom, Arch1
- `{Target}`: Door, Plant(Only in RealisticRoom), Coach(Only in RealisticRoom)
- `{Version}`: v0~v2, three different rules to sample start location.

Example Environments:
- UnrealSearch-RealisticRoomDoor-DiscreteColor-v0
- UnrealSearch-RealisticRoomCoach-DiscreteColor-v0
- UnrealSearch-RealisticRoomPlant-DiscreteColor-v0
- UnrealSearch-Arch1Door-DiscreteColor-v0

## Robotic Arm Control
Goal: move the arm to reach a goal position.

Naming rule: `UnrealArm-{ActionSpace}{ObsSpace}-{Versrion}`.
Specifically, these variables could be:
- `{ActionSpace}`: Discrete, Continuous
- `{ObsSpace}`: **Pose**, Color, Depth, Rgbd
- `{Version}`: v0 (Sample goals Randomly), v1 (Sample goals from a list in order)

An example usage is [Zuo et al., 2019](https://arxiv.org/abs/1812.00725), which use ``UnrealArm-ContinuousPose-vo`` to 
train a DDPG controller for robotic arm control.

If you use these robotic arm environments in your research work, we would be grateful if you could cite this paper:
```
@inproceedings{zuo2019craves,
  title={CRAVES: Controlling Robotic Arm with a Vision-based Economic System},
  author={Zuo, Yiming and Qiu, Weichao and Xie, Lingxi and Zhong, Fangwei and Wang, Yizhou and Yuille, Alan L},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4214--4223},
  year={2019}
}
```

## Active Object Tracking
Goal: follow a target object by autonomously controlling the motion system of a tracker given visual observations.

These active tracking environments are used in [Luo et al., 2018](https://arxiv.org/abs/1705.10561), 
[Luo et al., 2019](https://arxiv.org/abs/1808.03405), [Zhong et al., 2019](https://openreview.net/pdf?id=HkgYmhR9KX), [Zhong et al., 2019](https://openreview.net/pdf?id=HkgYmhR9KX).

## Cities
The `City1` and `City2` Environments are used in [Luo et al., 2018](https://arxiv.org/abs/1705.10561).
The target will move along a pre-defined trajectory. 

Binaries: [SplineCharacterA](https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/SplineCharacterA.zip), [SplineCharacterF](https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/SplineCharacterF.zip)

Naming rule: `UnrealTrack-{Scene}{Target}{PathID}-{ActionSpace}{ObsSpace}-{Versrion}`.

Specifically, these variables could be:
- `{Scene}{Target}{PathID}`: City1StefaniPath1, City2MalcomPath2
- `{ActionSpace}`: Discrete, Continuous
- `{ObsSpace}`: Color, Depth, Rgbd
- `{Version}`: v0 (Testing environment), v1 (Augmented environment for training)

In [Luo et al., 2018](https://arxiv.org/abs/1705.10561), `UnrealTrack-City1StefaniPath1-DiscreteColor-v1` is used for training, 
`UnrealTrack-City1StefaniPath1-DiscreteColor-v0` and `UnrealTrack-City2MalcomPath2-DiscreteColor-v0` are used for evaluation.

## More Advanced Environment Augmentation

**RandomRoom** is built for learning a generalizable tracker by more advanced environment augmentation techniques.
The illumination(color, direction, intensity), backgrounds(textures, roughness), the target(trajectory, appearance, speed) could be randomized in the Room. 

Binaries: [RandomRoom](https://www.cs.jhu.edu/~qiuwch/unrealcv/binaries/RandomRoom.zip)

Naming rule: `UnrealTrack-RandoomRoom-{ActionSpace}{ObsSpace}-{Versrion}`.
- `{Version}`: v0~v4, different levels of environment augmentation, v4 is used in [Luo et al., 2019](https://arxiv.org/abs/1808.03405).

Note that you need prepare [Textures](https://www.cs.jhu.edu/~qiuwch/unrealcv/binaries/Textures.zip) for background randomization, just running:
```
python load_env.py -e Textures
```

## Controllable Target
In these environments, the target each player is allowed to controlled by external program.
So you can design different rules to control the movement of target.

Binaries: [DuelingRoom](https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/DuelingRoom.zip),
[UrbanCity](https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/UrbanCity_2P.zip),
[SnowForest](https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/SnowForest_2P.zip),
[Garage](https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/Garage_2P.zip)

Naming rule: `UnrealTrack-{Scene}{Target}-{ActionSpace}{ObsSpace}-{Versrion}`.
- `{Scene}`: DuelingRoom, UrbanCity, Garage, SnowForest
- `{Target}`: PZR, Adv, Ram, Nav, NavShort, Internal

`PZR`, `Adv` is used for learning an adversarial target under different reward structure.
`Ram`, `Nav`, `NavShort`, `Internal` are four different rules to control the target. 

An example usage is [Zhong et al., 2019](https://openreview.net/pdf?id=HkgYmhR9KX), 
which formulate a multi-agent adversarial game between target and tracker to learn a stronger tracker in ``DuelingRoom``.  `UrbanCity`, `Garage`, `SnowForest` are built for evaluating the generalization of the learned tracker.

If you use these active tracking environments in your research work, we would be grateful if you could cite them as follow:
```
@inproceedings{luo2018end,
  title={End-to-end Active Object Tracking via Reinforcement Learning},
  author={Luo, Wenhan and Sun, Peng and Zhong, Fangwei and Liu, Wei and Zhang, Tong and Wang, Yizhou},
  booktitle={International Conference on Machine Learning},
  pages = {3286--3295},
  year = {2018}
  }

@article{luo2019end,
  title={End-to-end active object tracking and its real-world deployment via reinforcement learning},
  author={Luo, Wenhan and Sun, Peng and Zhong, Fangwei and Liu, Wei and Zhang, Tong and Wang, Yizhou},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={42},
  number={6},
  pages={1317--1332},
  year={2019},
  publisher={IEEE}
}
 
@inproceedings{zhong2018advat,
  title={AD-VAT: An asymmetric dueling mechanism for learning visual active tracking},
  author={Zhong, Fangwei and Sun, Peng and Luo, Wenhan and Yan, Tingyun and Wang, Yizhou},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=HkgYmhR9KX},
  }

  @article{zhong2021advat,
  title={AD-VAT+: An Asymmetric Dueling Mechanism for Learning and Understanding Visual Active Tracking},
  author={Zhong, Fangwei and Sun, Peng and Luo, Wenhan and Yan, Tingyun and Wang, Yizhou},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  year={2019},
  volume={43},
  number={5},
  pages={1467-1482}
  }

@inproceedings{li2020pose,
  title={Pose-assisted multi-camera collaboration for active object tracking},
  author={Li, Jing and Xu, Jing and Zhong, Fangwei and Kong, Xiangyu and Qiao, Yu and Wang, Yizhou},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2020},
  volume={34},
  number={01},
  pages={759--766}
}

@InProceedings{zhong2021distraction,
title = {Towards Distraction-Robust Active Visual Tracking},
author = {Zhong, Fangwei and Sun, Peng and Luo, Wenhan and Yan, Tingyun and Wang, Yizhou}, 
booktitle = {Proceedings of the 38th International Conference on Machine Learning}, 
year = {2021},
volume = {139},
pages = {12782--12792}
}
```