from gym.envs.registration import register
import logging
logger = logging.getLogger(__name__)
use_docker = True  # True: use nvidia docker   False: do not use nvidia-docker



# RrDoor41
register(
    id='Search-RrDoorDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_door41.json',
              'reset_type' : 'waypoint',
              'test': False,
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type': 'bbox',
              'docker': use_docker
              },
    max_episode_steps = 1000000

)



register(
    id='Search-RrDoorContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_door41.json',
              'reset_type' : 'waypoint',
              'test': False,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type': 'bbox',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)
register(
    id='Search-RrDoorDiscreteTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_door41.json',
              'reset_type': 'testpoint',
              'test': True,
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type': 'bbox',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)
register(
    id='Search-RrDoorContinuousTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_door41.json',
              'reset_type': 'testpoint',
              'test': True,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type': 'bbox',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)


#Arch1Door1
register(
    id='Search-Arch1DoorDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door1.json',
              'reset_type': 'waypoint',
              'test': False,
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type': 'bbox',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)
register(
    id='Search-Arch1DoorContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door1.json',
              'reset_type': 'waypoint',
              'test': False,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type': 'bbox',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)
register(
    id='Search-Arch1DoorDiscreteTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door1.json',
              'reset_type': 'testpoint',
              'test': True,
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type': 'bbox',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)
register(
    id='Search-Arch1DoorContinuousTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door1.json',
              'reset_type': 'testpoint',
              'test': True,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type': 'bbox',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)


# RrMultiPlants  Finding plants(large objects)
register(
    id='Search-RrPlantsDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant78.json',
              'reset_type': 'waypoint',
              'test': False,
              'action_type' : 'discrete',
              'observation_type': 'rgbd',
              'reward_type': 'bbox',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)
register(
    id='Search-RrPlantsContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant78.json',
              'reset_type': 'waypoint',
              'test': False,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)

register(
    id='Search-RrPlantsDiscreteTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant78.json',
              'reset_type': 'testpoint',
              'test': True,
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type': 'bbox',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)
register(
    id='Search-RrPlantsContinuousTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant78.json',
              'reset_type': 'testpoint',
              'test': True,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type': 'bbox',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)


# RrMultiSockets, finding sockets(small object)
register(
    id='Search-RrSocketsDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_socket.json',
              'reset_type': 'waypoint',
              'test': False,
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type': 'bbox',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)
register(
    id='Search-RrSocketsContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_socket.json',
              'reset_type': 'waypoint',
              'test': False,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)


register(
    id='Search-LoftSofaDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_loft_sofa.json',
              'augment_env' : 'texture',
              'reset_type' : 'waypoint',
              'test': False,
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type': 'bbox',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)

register(
    id='Search-LoftPlantDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_topview',
    kwargs = {'setting_file' : 'search_loft_plant.json',
              'reset_type' : 'random',
              'test': True,
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type': 'distance',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)

register(
    id='Search-ForestDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_topview',
    kwargs = {'setting_file' : 'search_tree.json',
              'reset_type' : 'random',
              'test': True,
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type': 'distance',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)

register(
    id='RobotArm-Discrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvRobotArm_base',
    kwargs = {'setting_file' : 'robotarm_v2.json',
              'reset_type': 'keyboard',
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type': 'move_distance',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)

register(
    id='RobotArm-Discrete-v1',
    entry_point='gym_unrealcv.envs:UnrealCvRobotArm_base',
    kwargs = {'setting_file' : 'robotarm_v2.json',
              'reset_type': 'keyboard',
              'action_type' : 'discrete',
              'observation_type': 'measured',
              'reward_type': 'move_distance',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)

register(
    id='RobotArm-Continuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvRobotArm_base',
    kwargs = {'setting_file' : 'robotarm_v2.json',
              'reset_type': 'keyboard',
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type': 'move_distance',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)

register(
    id='UAV-Discrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvQuadcopter_base',
    kwargs = {'setting_file' : 'search_quadcopter1.json',
              'reset_type': 'waypoint',
              'test': False,
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type': 'bbox',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)

#A
register(
    id='Tracking-City2MalcomPath2Static-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base',
    kwargs = {'setting_file' : 'tracking_v0.4_A.json',
              'reset_type': 'static_hide',
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)

#B
register(
    id='Tracking-B-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base',
    kwargs = {'setting_file' : 'tracking_v0.4_B.json',
              'reset_type': 'static',
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)
#C
register(
    id='Tracking-C-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base',
    kwargs = {'setting_file' : 'tracking_v0.4_C.json',
              'reset_type': 'static',
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)
#D
register(
    id='Tracking-D-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base',
    kwargs = {'setting_file' : 'tracking_v0.4_D.json',
              'reset_type': 'static',
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)
#E
register(
    id='Tracking-City1MalcomStatic-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base',
    kwargs = {'setting_file' : 'tracking_v0.4_E.json',
              'reset_type': 'static',
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)


#F
register(
    id='Tracking-City1StefaniPath1Static-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base',
    kwargs = {'setting_file' : 'tracking_v0.4_F.json',
              'reset_type': 'static',
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)

#Ftrain
register(
    id='Tracking-City1StefaniPath1Random-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base',
    kwargs = {'setting_file' : 'tracking_v0.4_F.json',
              'reset_type': 'random',
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)

#G
register(
    id='Tracking-City1StefaniPath2Static-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base',
    kwargs = {'setting_file' : 'tracking_v0.4_G.json',
              'reset_type': 'static',
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)