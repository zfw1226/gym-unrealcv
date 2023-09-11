import os
import numpy as np


def load_env_setting(filename):
    f = open(get_settingpath(filename))
    type = os.path.splitext(filename)[1]
    if type == '.json':
        import json
        setting = json.load(f)
    else:
        print ('unknown type')
    return setting


def get_settingpath(filename):
    import gym_unrealcv
    gympath = os.path.dirname(gym_unrealcv.__file__)
    return os.path.join(gympath, 'envs/setting', filename)


def get_action_size(action):
    return len(action)


def get_direction(current_pose, target_pose):  # get relative angle between current pose and target pose in x-y plane
    y_delt = target_pose[1] - current_pose[1]
    x_delt = target_pose[0] - current_pose[0]
    if x_delt == 0 and y_delt == 0:  # if the same position
        return 0
    angle_abs = np.arctan2(y_delt, x_delt)/np.pi*180
    angle_relative = angle_abs - current_pose[4]
    if angle_relative > 180:
        angle_relative -= 360
    if angle_relative < -180:
        angle_relative += 360
    return angle_relative


def get_textures(texture_dir, docker):
    import gym_unrealcv
    gym_path = os.path.dirname(gym_unrealcv.__file__)
    texture_dir = os.path.join(gym_path, 'envs', 'UnrealEnv', texture_dir)
    textures_list = os.listdir(texture_dir)
    # relative to abs
    for i in range(len(textures_list)):
        if docker:
            textures_list[i] = os.path.join('/unreal', texture_dir, textures_list[i])
        else:
            textures_list[i] = os.path.join(texture_dir, textures_list[i])
    return textures_list

def convert_dict(old_dict):
    new_dict = {}
    for agent, info in old_dict.items():
        names = info["name"]
        for i, name in enumerate(names):
            new_dict[name] = {
                "cam_id": info["cam_id"][i],
                "internal_nav": info["internal_nav"],
                "agent_type": agent,
                "discrete_action": info["discrete_action"],
                "continuous_action": info["continuous_action"],
                "class_name": info["class_name"][i],
                "relative_location": info["relative_location"],
                "relative_rotation": info["relative_rotation"]
            }
    return new_dict