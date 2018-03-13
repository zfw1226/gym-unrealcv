from gym.envs.registration import register
import logging
logger = logging.getLogger(__name__)
use_docker = True  # True: use nvidia docker   False: do not use nvidia-docker



register(
    id='Search-BProom-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_3d',
    kwargs = {'setting_file' : 'search_bproom.json',
              'reset_type' : 'random',
              'test': False,
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type': 'bbox_distance',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)

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
              'observation_type': 'color',
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
    kwargs = {'setting_file' : 'robotarm_v3.json',
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
    kwargs = {'setting_file' : 'robotarm_v4.json',
              'reset_type': 'keyboard',
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type': 'move_distance',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)

register(
    id='RobotArm-Continuous-v1',
    entry_point='gym_unrealcv.envs:UnrealCvRobotArm_base',
    kwargs = {'setting_file' : 'robotarm_v3.json',
              'reset_type': 'keyboard',
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type': 'move_distance',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)



register(
    id='RobotArm-Continuous-v2',
    entry_point='gym_unrealcv.envs:UnrealCvRobotArm_base',
    kwargs = {'setting_file' : 'robotarm_v4.json',
              'reset_type': 'keyboard',
              'action_type' : 'continuous',
              'observation_type': 'measured',
              'reward_type': 'move_distance',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)

register(
    id='RobotArm-Continuous-v3',
    entry_point='gym_unrealcv.envs:UnrealCvRobotArm_base',
    kwargs = {'setting_file' : 'robotarm_v4.json',
              'reset_type': 'keyboard',
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type': 'move_distance',
              'docker': use_docker
              },
    max_episode_steps = 1000000
)

register(
    id='RobotArm-Continuous-v4',
    entry_point='gym_unrealcv.envs:UnrealCvRobotArm_base',
    kwargs = {'setting_file' : 'robotarm_v5.json',
              'reset_type': 'keyboard',
              'action_type' : 'continuous',
              'observation_type': 'measured_real',
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

register(
    id='Tracking-City2MalcomPath2StaticContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base',
    kwargs = {'setting_file' : 'tracking_v0.4_A.json',
              'reset_type': 'static_hide',
              'action_type' : 'continuous',
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

register(
    id='Tracking-City1StefaniPath1RandomContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base',
    kwargs = {'setting_file' : 'tracking_v0.4_F.json',
              'reset_type': 'random',
              'action_type' : 'continuous',
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

register(
    id='Tracking-City1StefaniPath2StaticContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base',
    kwargs = {'setting_file' : 'tracking_v0.4_G.json',
              'reset_type': 'static',
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)

register(
    id='Tracking-Train-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
    kwargs = {'setting_file' : 'tracking_train_random_v0.1.json',
              'reset_type': 'random',
              'action_type' : 'discrete',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)

register(
    id='Tracking-TrainContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
    kwargs = {'setting_file' : 'tracking_train_random_v0.1.json',
              'reset_type': 'random',
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)

# simple
register(
    id='Tracking-TrainContinuous-v1',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
    kwargs = {'setting_file' : 'tracking_train_random_v0.1.json',
              'reset_type': 'static',
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)

register(
    id='Tracking-TrainContinuous-v2',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
    kwargs = {'setting_file' : 'tracking_train_random_v0.2.json',
              'reset_type': 'random',
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)

register(
    id='Tracking-TestDesert-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
    kwargs = {'setting_file' : 'tracking_test_desert_v0.1.json',
              'reset_type': 'static',
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)


register(
    id='Tracking-TrainUrban-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
    kwargs = {'setting_file' : 'tracking_test_city_v0.1.json',
              'reset_type': 'static',
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)

register(
    id='Tracking-TestUrban-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
    kwargs = {'setting_file' : 'tracking_test_city_v0.1.json',
              'reset_type': 0,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)



register(
    id='Tracking-TestRoom-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
    kwargs = {'setting_file' : 'tracking_test_room_v0.1.json',
              'reset_type': 'static',
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)

register(
    id='Tracking-TrainRoom-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
    kwargs = {'setting_file' : 'tracking_train_room_v0.1.json',
              'reset_type': 0,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)

register(
    id='Tracking-TrainRoom-v1',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
    kwargs = {'setting_file' : 'tracking_train_room_v0.0.json',
              'reset_type': 1,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)

register(
    id='Tracking-TrainRoom-v2',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
    kwargs = {'setting_file' : 'tracking_train_room_v0.0.json',
              'reset_type': 2,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)

register(
    id='Tracking-TrainRoom-v3',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
    kwargs = {'setting_file' : 'tracking_train_room_v0.0.json',
              'reset_type': 3,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)

register(
    id='Tracking-TrainRoom-v4',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
    kwargs = {'setting_file' : 'tracking_train_room_v0.0.json',
              'reset_type': 4,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)

register(
    id='Tracking-TrainRoom-v5',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
    kwargs = {'setting_file' : 'tracking_train_room_v0.0.json',
              'reset_type': 5,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)

register(
    id='Tracking-TestArch-v1',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
    kwargs = {'setting_file' : 'tracking_test_arch_v1.json',
              'reset_type': 0,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)

register(
    id='Tracking-TestArch-v2',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
    kwargs = {'setting_file' : 'tracking_test_arch_v2.json',
              'reset_type': 0,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)

register(
    id='Tracking-TestArch-v3',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
    kwargs = {'setting_file' : 'tracking_test_arch_v3.json',
              'reset_type': 0,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)

register(
    id='Tracking-TestArch-v4',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
    kwargs = {'setting_file' : 'tracking_test_arch_v4.json',
              'reset_type': 0,
              'action_type' : 'continuous',
              'observation_type': 'color',
              'reward_type':  'distance',
              'docker': use_docker,
              },
    max_episode_steps = 1000000
)




register(
    id='Tracking-Video-v0',
    entry_point='gym_unrealcv.envs:VideoTracking_base',
    max_episode_steps = 1000000
)