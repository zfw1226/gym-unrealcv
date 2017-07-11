from gym.envs.registration import register
import logging
logger = logging.getLogger(__name__)
use_docker = True  # True: use nvidia docker   False: do not use nvidia-docker



# RrDoor41
register(
    id='Search-RrDoorDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_door41.json',
              'test': False,
              'action_type' : 'discrete',
              'observation_type': 'rgbd',
              'reward_type': 'bbox',
              'docker': use_docker
              }
)
register(
    id='Search-RrDoorContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_door41.json',
              'test': False,
              'action_type' : 'continuous',
              'observation_type': 'rgbd',
              'reward_type': 'bbox',
              'docker': use_docker
              }
)
register(
    id='Search-RrDoorDiscreteTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_door41.json',
              'test': True,
              'action_type' : 'discrete',
              'observation_type': 'rgbd',
              'reward_type': 'bbox',
              'docker': use_docker

              }
)
register(
    id='Search-RrDoorContinuousTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_door41.json',
              'test': True,
              'action_type' : 'continuous',
              'observation_type': 'rgbd',
              'reward_type': 'bbox',
              'docker': use_docker
              }
)


#Arch1Door1
register(
    id='Search-Arch1DoorDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door1.json',
              'test': False,
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type': 'bbox',
              'docker': use_docker
              }
)
register(
    id='Search-Arch1DoorContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door1.json',
              'test': False,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type': 'bbox',
              'docker': use_docker
              }
)
register(
    id='Search-Arch1DoorDiscreteTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door1.json',
              'test': True,
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type': 'bbox',
              'docker': use_docker
              }
)
register(
    id='Search-Arch1DoorContinuousTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door12.json',
              'test': True,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'docker': use_docker
              }
)


# RrMultiPlants
register(
    id='Search-RrMultiPlantsDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant78.json',
              'test': False,
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type': 'bbox',
              'docker': use_docker
              }
)
register(
    id='Search-RrMultiPlantsContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant78.json',
              'test': False,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'docker': use_docker
              }
)

register(
    id='Search-RrMultiPlantsDiscreteTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant78.json',
              'test': True,
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type': 'bbox',
              'docker': use_docker
              },
)
register(
    id='Search-RrMultiPlantsContinuousTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant78.json',
              'test': True,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type': 'bbox',
              'docker': use_docker
              },
)
