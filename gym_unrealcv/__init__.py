from gym.envs.registration import register
import logging
logger = logging.getLogger(__name__)
use_docker = False  # True: use nvidia docker   False: do not use nvidia-docker

#RrPlant7
register(
    id='Search-RrPlantDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant7.yaml',
              'test' : False,
              'action_type' : 'discrete',
              'observation_type' : 'rgbd',
              'docker' : use_docker
              }
)
register(
    id='Search-RrPlantContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant7.yaml',
              'test' : False,
              'action_type' : 'continuous',
              'observation_type' : 'rgbd',
              'docker' : use_docker
              }
)

register(
    id='Search-RrPlantDiscreteTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant7.yaml',
              'test' : True,
              'action_type' : 'discrete',
              'observation_type' : 'rgbd',
              'docker': use_docker
              }
)
register(
    id='Search-RrPlantContinuousTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant7.yaml',
              'test' : True,
              'action_type' : 'continuous',
              'observation_type' : 'rgbd',
              'docker': use_docker
              }
)

# RrPlant8
register(
    id='Search-RrPlantDiscrete-v1',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant8.yaml',
              'test': False,
              'action_type' : 'discrete',
              'observation_type': 'rgbd',
              'docker': use_docker
              }
)
register(
    id='Search-RrPlantContinuous-v1',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant8.yaml',
              'test': False,
              'action_type' : 'continuous',
              'observation_type': 'rgbd',
              'docker': use_docker
              }
)
register(
    id='Search-RrPlantDiscreteTest-v1',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant8.yaml',
              'test': True,
              'action_type' : 'discrete',
              'observation_type': 'rgbd',
              'docker': use_docker
              }
)
register(
    id='Search-RrPlantContinuousTest-v1',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant8.yaml',
              'test': True,
              'action_type' : 'continuous',
              'observation_type': 'rgbd',
              'docker': use_docker
              }
)

# RrDoor41
register(
    id='Search-RrDoorDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_door41.yaml',
              'test': False,
              'action_type' : 'discrete',
              'observation_type': 'rgbd',
              'docker': use_docker
              }
)
register(
    id='Search-RrDoorContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_door41.yaml',
              'test': False,
              'action_type' : 'continuous',
              'observation_type': 'rgbd',
              'docker': use_docker
              }
)
register(
    id='Search-RrDoorDiscreteTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_door41.yaml',
              'test': True,
              'action_type' : 'discrete',
              'observation_type': 'rgbd'
              }
)
register(
    id='Search-RrDoorContinuousTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_door41.yaml',
              'test': True,
              'action_type' : 'continuous',
              'observation_type': 'rgbd',
              'docker': use_docker
              }
)


#Arch1Door1
register(
    id='Search-Arch1DoorDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door1.yaml',
              'test': False,
              'action_type' : 'discrete',
              'observation_type': 'rgbd',
              'docker': use_docker
              }
)
register(
    id='Search-Arch1DoorContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door1.yaml',
              'test': False,
              'action_type' : 'continuous',
              'observation_type': 'rgbd',
              'docker': use_docker
              }
)
register(
    id='Search-Arch1DoorDiscreteTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door1.yaml',
              'test': True,
              'action_type' : 'discrete',
              'observation_type': 'rgbd',
              'docker': use_docker
              }
)
register(
    id='Search-Arch1DoorContinuousTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door12.yaml',
              'test': True,
              'action_type' : 'continuous',
              'observation_type': 'rgbd',
              'docker': use_docker
              }
)


# RrMultiPlants
register(
    id='Search-RrMultiPlantsDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant78.yaml',
              'test': False,
              'action_type' : 'discrete',
              'observation_type': 'rgbd',
              'docker': use_docker
              }
)
register(
    id='Search-RrMultiPlantsContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant78.yaml',
              'test': False,
              'action_type' : 'continuous',
              'observation_type': 'rgbd',
              'docker': use_docker
              }
)

register(
    id='Search-RrMultiPlantsDiscreteTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant78.yaml',
              'test': True,
              'action_type' : 'discrete',
              'observation_type': 'rgbd',
              'docker': use_docker
              },
)
register(
    id='Search-RrMultiPlantsContinuousTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant78.yaml',
              'test': True,
              'action_type' : 'continuous',
              'observation_type': 'rgbd',
              'docker': use_docker
              },
)


# Arch1MultiDoors
register(
    id='Search-Arch1MultiDoorsDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door12.yaml',
              'test': False,
              'action_type' : 'discrete',
              'observation_type': 'rgbd',
              'docker': use_docker
              }
)
register(
    id='Search-Arch1MultiDoorsContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door12.yaml',
              'test': False,
              'action_type' : 'continuous',
              'observation_type': 'rgbd',
              'docker': use_docker
              }
)
register(
    id='Search-Arch1MultiDoorsDiscreteTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door12.yaml',
              'test': True,
              'action_type' : 'discrete',
              'observation_type': 'rgbd',
              'docker': use_docker
              }
)
register(
    id='Search-Arch1MultiDoorsContinuousTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door12.yaml',
              'test': True,
              'action_type' : 'continuous',
              'observation_type': 'rgbd',
              'docker': use_docker
              }
)