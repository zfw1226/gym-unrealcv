from gym.envs.registration import register
import logging
from gym_unrealcv.envs.utils.misc import load_env_setting
logger = logging.getLogger(__name__)
use_docker = False  # True: use nvidia docker   False: do not use nvidia-docker

# Searching/Navigation
# -------------------------------------------------
for env in ['RealisticRoom', 'Arch1']:
    setting_file = 'searching/{env}.json'.format(env=env)
    settings = load_env_setting(setting_file)
    for i, reset in enumerate(['random', 'waypoint', 'testpoint']):
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd']:  # observation type
                for category in settings['targets']:
                    register(
                        id='UnrealSearch-{env}{category}-{action}{obs}-v{reset}'.format(env=env, category=category, action=action, obs=obs, reset=i),
                        entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
                        kwargs={'setting_file': 'searching/{env}.json'.format(env=env),
                                'category': category,
                                'reset_type': reset,
                                'action_type': action,
                                'observation_type': obs,
                                'reward_type': 'bbox_distance',  # bbox, distance, bbox_distance
                                'docker': use_docker,
                                },
                        max_episode_steps=200
                    )

# ------------------------------------------------------------------
# Robot Arm
# "CRAVES: Controlling Robotic Arm With a Vision-Based Economic System", CVPR 2019
for action in ['Discrete', 'Continuous']:  # action type
    for obs in ['Pose', 'Color', 'Depth', 'Rgbd']:
        for i in range(3):
            register(
                    id='UnrealArm-{action}{obs}-v{version}'.format(action=action, obs=obs, version=i),
                    entry_point='gym_unrealcv.envs:UnrealCvRobotArm_reach',
                    kwargs={'setting_file': 'robotarm/robotarm_reach.json',
                            'action_type': action,
                            'observation_type': obs,
                            'docker': use_docker,
                            'version': i
                            },
                    max_episode_steps=100
                        )

# -----------------------------------------------------------------------
# Tracking
# "End-to-end Active Object Tracking via Reinforcement Learning", ICML 2018
for env in ['City1', 'City2']:
    for target in ['Malcom', 'Stefani']:
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd']:  # observation type
                for path in ['Path1', 'Path2']:  # observation type
                    for i, reset in enumerate(['Static', 'Random']):
                        register(
                            id='UnrealTrack-{env}{target}{path}-'
                               '{action}{obs}-v{reset}'.format(env=env, target=target, path=path,
                                                               action=action, obs=obs, reset=i),
                            entry_point='gym_unrealcv.envs:UnrealCvTracking_spline',
                            kwargs={'setting_file': 'tracking_v0/{env}{target}{path}.json'.format(
                                env=env, target=target, path=path),
                                    'reset_type': reset,
                                    'action_type': action,
                                    'observation_type': obs,
                                    'reward_type': 'distance',
                                    'docker': use_docker,
                                    },
                            max_episode_steps=3000
                            )

# "End-to-end Active Object Tracking and Its Real-world Deployment via Reinforcement Learning", IEEE TPAMI
for env in ['RandomRoom']:
    for i in range(5):  # reset type
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd']:  # observation type
                register(
                    id='UnrealTrack-{env}-{action}{obs}-v{reset}'.format(env=env, action=action, obs=obs, reset=i),
                    entry_point='gym_unrealcv.envs:UnrealCvTracking_random',
                    kwargs={'setting_file': 'tracking_v0/{env}.json'.format(env=env),
                            'reset_type': i,
                            'action_type': action,
                            'observation_type': obs,
                            'reward_type': 'distance',
                            'docker': use_docker,
                            },
                    max_episode_steps=500
                )


# "AD-VAT: An Asymmetric Dueling mechanism for learning Visual Active Tracking", ICLR 2019
# DuelingRoom is the training environment, others are testing environment.
for env in ['DuelingRoom', 'UrbanCity', 'UrbanRoad', 'Garage', 'SnowForest', 'Forest', 'Garden']:
    for i in range(6):  # reset type
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd']:  # observation type
                for target in ['Ram', 'Nav', 'NavBase', 'NavShort', 'NavFix', 'Internal', 'PZR', 'Adv']:
                    name = 'UnrealTrack-{env}{target}-{action}{obs}-v{reset}'.format(
                        env=env, action=action, obs=obs, target=target, reset=i)
                    setting_file = 'tracking_1v1/{env}.json'.format(env=env)
                    register(
                        id=name,
                        entry_point='gym_unrealcv.envs:UnrealCvTracking_1v1',
                        kwargs={'setting_file': setting_file,
                                'reset_type': i,
                                'action_type': action,
                                'observation_type': obs,
                                'reward_type': 'distance',
                                'docker': use_docker,
                                'target': target
                                },
                        max_episode_steps=500
                    )


# "Pose-Assisted Multi-Camera Collaboration for Active Object Tracking", AAAI 2020
for env in ['MCRoom', 'Garden', 'UrbanTree']:
    for i in range(7):  # reset type
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd', 'Gray']:  # observation type
                for nav in ['Random', 'Goal', 'Internal', 'None',
                            'RandomInterval', 'GoalInterval', 'InternalInterval', 'NoneInterval']:
                    name = 'Unreal{env}-{action}{obs}{nav}-v{reset}'.format(env=env, action=action, obs=obs, nav=nav, reset=i)
                    setting_file = 'tracking_multicam/{env}.json'.format(env=env)
                    register(
                        id=name,
                        entry_point='gym_unrealcv.envs:UnrealCvMC',
                        kwargs={'setting_file': setting_file,
                                'reset_type': i,
                                'action_type': action,
                                'observation_type': obs,
                                'reward_type': 'distance',
                                'docker': use_docker,
                                'nav': nav
                                },
                        max_episode_steps=500
                    )
