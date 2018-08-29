from gym.envs.registration import register
import logging
logger = logging.getLogger(__name__)
use_docker = False  # True: use nvidia docker   False: do not use nvidia-docker


def load_env_setting(filename):
    import os
    import gym_unrealcv
    gympath = os.path.dirname(gym_unrealcv.__file__)
    gympath = os.path.join(gympath, 'envs/setting', filename)
    f = open(gympath)
    filetype = os.path.splitext(filename)[1]
    if filetype == '.json':
        import json
        setting = json.load(f)
    else:
        print ('unknown type')

    return setting
# Searching/Navigation
# -------------------------------------------------
for env in ['BpRoom']:
    for i, reset in enumerate(['random', 'testpoint']):
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd']:  # observation type
                register(
                    id='Search{env}-{action}{obs}-v{reset}'.format(env=env, action=action, obs=obs, reset=i),
                    entry_point='gym_unrealcv.envs:UnrealCvSearch_3d',
                    kwargs={'setting_file': 'searching/{env}.json'.format(env=env),
                            'reset_type': reset,
                            'test': False,
                            'action_type': action,
                            'observation_type': obs,
                            'reward_type': 'bbox_distance',
                            'docker': use_docker,
                            },
                    max_episode_steps=200
                )

for env in ['RealisticRoom', 'Arch1', 'Arch2']:
    setting_file = 'searching/{env}.json'.format(env=env)
    settings = load_env_setting(setting_file)
    for i, reset in enumerate(['random', 'waypoint', 'testpoint']):
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd']:  # observation type
                for category in settings['targets']:
                    register(
                        id='Search{env}{category}-{action}{obs}-v{reset}'.format(env=env, category=category, action=action, obs=obs, reset=i),
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

# Robot Arm
# ------------------------------------------------------------------
for action in ['Discrete', 'Continuous']:  # action type
    for obs in ['Color', 'Depth', 'Rgbd', 'Measured', 'MeasuredQR']:
        for i, reward in enumerate(['distance', 'move', 'move_distance']):
            register(
                id='RobotArm-{action}{obs}-v{reward}'.format(action=action, obs=obs, reward=i),
                entry_point='gym_unrealcv.envs:UnrealCvRobotArm_base',
                kwargs={'setting_file': 'robotarm/robotarm_v0.json',
                        'reset_type': 'keyboard',
                        'action_type': action,
                        'observation_type': obs,
                        'reward_type': reward,
                        'docker': use_docker
                        },
                max_episode_steps=1000000
            )

# Tracking
# -----------------------------------------------------------------------
# old env
for env in ['City1', 'City2']:
    for target in ['Malcom', 'Stefani']:
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd']:  # observation type
                for path in ['Path1', 'Path2']:  # observation type
                    for i, reset in enumerate(['static', 'random']):
                        register(
                            id='Tracking{env}{target}{path}-{action}{obs}-v{reset}'.format(env=env, target=target, path=path,
                                                                                     action=action, obs=obs, reset=i),
                            entry_point='gym_unrealcv.envs:UnrealCvTracking_base',
                            kwargs={'setting_file': 'tracking_v0/{env}{target}{path}.json'.format(env=env, target=target, path=path),
                                    'reset_type': reset,
                                    'action_type': action,
                                    'observation_type': obs,
                                    'reward_type': 'distance',
                                    'docker': use_docker,
                                    },
                            max_episode_steps=1000
                            )

# new training env
for env in ['SimpleRoom', 'BpRoom', 'MetaRoom']:
    for i in range(5):  # reset type
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd']:  # observation type
                register(
                    id='Tracking{env}-{action}{obs}-v{reset}'.format(env=env, action=action, obs=obs, reset=i),
                    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
                    kwargs={'setting_file': 'tracking_v1/{env}.json'.format(env=env),
                            'reset_type': i,
                            'action_type': action,
                            'observation_type': obs,
                            'reward_type': 'distance',
                            'docker': use_docker,
                            },
                    max_episode_steps=1000
                )

# new testing
for env in ['UrbanCity', 'Arch1', 'Arch2', 'Arch3']:
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd']:  # observation type
                register(
                    id='Tracking{env}-{action}{obs}Test-v{reset}'.format(env=env, action=action, obs=obs, reset=0),
                    entry_point='gym_unrealcv.envs:UnrealCvTracking_base_random',
                    kwargs={'setting_file': 'tracking_v1/{env}.json'.format(env=env),
                            'reset_type': -1,
                            'action_type': action,
                            'observation_type': obs,
                            'reward_type': 'distance',
                            'docker': use_docker,
                            },
                    max_episode_steps=1000
                )

# new training env
for env in ['MPRoom']:
    for i in range(5):  # reset type
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd', 'Gray']:  # observation type
                for nav in ['Random', 'Goal', 'Internal', False]:
                    if nav:
                        name = 'UnrealTracking{env}-{action}{obs}{nav}-v{reset}'.format(env=env, action=action, obs=obs, nav=nav, reset=i)
                    else:
                        name = 'UnrealTracking{env}-{action}{obs}-v{reset}'.format(env=env, action=action, obs=obs, reset=i)
                    register(
                        id=name,
                        entry_point='gym_unrealcv.envs:UnrealCvTracking_multi',
                        kwargs={'setting_file': 'tracking_multi/{env}.json'.format(env=env),
                                'reset_type': i,
                                'action_type': action,
                                'observation_type': obs,
                                'reward_type': 'distance',
                                'docker': use_docker,
                                'nav': nav
                                },
                        max_episode_steps=500
                    )
# test video image
register(
    id='Tracking-Video-v0',
    entry_point='gym_unrealcv.envs:VideoTracking_base',
    max_episode_steps=1000
)


register(
    id='Tracking-B-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base',
    kwargs={'setting_file': 'tracking_v0/tracking_v0.4_B.json',
            'reset_type': 'static',
            'action_type': 'discrete',
            'observation_type': 'color',
            'reward_type':  'distance',
            'docker': use_docker,
            },
    max_episode_steps=1000
)
register(
    id='Tracking-C-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base',
    kwargs={'setting_file': 'tracking_v0/tracking_v0.4_C.json',
            'reset_type': 'static',
            'action_type': 'discrete',
            'observation_type': 'color',
            'reward_type':  'distance',
            'docker': use_docker,
            },
    max_episode_steps=1000
)
register(
    id='Tracking-D-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base',
    kwargs={'setting_file': 'tracking_v0/tracking_v0.4_D.json',
            'reset_type': 'static',
            'action_type': 'discrete',
            'observation_type': 'color',
            'reward_type':  'distance',
            'docker': use_docker,
            },
    max_episode_steps=1000
)
register(
    id='Tracking-City1MalcomStatic-v0',
    entry_point='gym_unrealcv.envs:UnrealCvTracking_base',
    kwargs={'setting_file': 'tracking_v0/tracking_v0.4_E.json',
            'reset_type': 'static',
            'action_type': 'discrete',
            'observation_type': 'color',
            'reward_type':  'distance',
            'docker': use_docker,
            },
    max_episode_steps=1000
)