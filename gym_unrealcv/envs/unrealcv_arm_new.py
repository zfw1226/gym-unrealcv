import math
import os
import random
import time

import gym
import numpy as np
from gym import spaces
from gym_unrealcv.envs.robotarm.visualization import show_info
from gym_unrealcv.envs.utils import env_unreal
from gym_unrealcv.envs.robotarm.interaction import Robotarm


class UnrealCvRobotArm_base(gym.Env):
    def __init__(self,
                 setting_file,
                 reset_type='keyboard',    # keyboard, bp
                 action_type='continuous',   # 'discrete', 'continuous'
                 observation_type='MeasuredReal',  # 'color', 'depth', 'rgbd' . 'measure'
                 reward_type='distance',  # distance, move, move_distance
                 docker=False,
                 resolution=(160, 120),
                 ):

        # load and process setting
        setting = self.load_env_setting(setting_file)
        self.cam_id = setting['cam_view_id']
        self.max_steps = setting['maxsteps']
        self.camera_pose = setting['camera_pose']
        self.discrete_actions = setting['discrete_actions']
        self.continous_actions = setting['continous_actions']
        self.pose_range = setting['pose_range']
        self.goal_range = setting['goal_range']
        self.env_bin = setting['env_bin']
        self.env_map = setting['env_map']
        self.objects = setting['objects']
        self.docker = docker
        self.reset_type = reset_type
        self.resolution = resolution

        # define reward type
        # distance, bbox, bbox_distance,
        self.reward_type = reward_type
        self.launched = False

        # define action type
        self.action_type = action_type
        if self.action_type == 'Discrete':
            self.action_space = spaces.Discrete(len(self.discrete_actions))
        elif self.action_type == 'Continuous':
            self.action_space = spaces.Box(low=np.array(self.continous_actions['low']),
                                           high=np.array(self.continous_actions['high']))

        self.pose_low = np.array(self.pose_range['low'])
        self.pose_high = np.array(self.pose_range['high'])

        self.count_steps = 0

        # define observation space,
        # color, depth, rgbd...
        self.launch_env()
        self.observation_type = observation_type
        self.observation_space = self.unrealcv.define_observation(self.cam_id, observation_type, setting)

    def launch_env(self):
        if self.launched:
            return True
        # start unreal env
        self.unreal = env_unreal.RunUnreal(ENV_BIN=self.env_bin, ENV_MAP=self.env_map)
        env_ip, env_port = self.unreal.start(self.docker, self.resolution)

        # connect UnrealCV
        self.unrealcv =Robotarm(cam_id=self.cam_id,
                                pose_range=self.pose_range,
                                port=env_port,
                                ip=env_ip,
                                targets=[],
                                env=self.unreal.path2env,
                                resolution=self.resolution)
        self.launched = True
        return self.launched

    def _step(self, action):
        info = dict(
            Done=False,
            Reward=0.0,
            Action=action,
            ArmPose=[],
            GripPosition=[],
            Steps=self.count_steps,
            TargetPose=self.goal_pos,
            Color=None,
            Depth=None,
        )
        action = np.squeeze(action)
        # take a action
        if self.action_type == 'Discrete':
            arm_state = self.unrealcv.move_arm(self.discrete_actions[action], mode='move')

        elif self.action_type == 'Continuous':
            arm_state = self.unrealcv.move_arm(np.append(action, 0), mode='move')

        self.count_steps += 1
        done = False
        tip_pose = self.unrealcv.get_tip_pose()
        tip_pos_tf = self.xyz2trz(tip_pose)
        distance = self.get_distance(self.goal_pos, tip_pos_tf, norm=True)

        # reward function
        reward = -0.1
        if arm_state:  # reach limitation
            done = True
            reward = -10
        elif distance < 0.05:  # reach
            reward = 1
            self.count_reach += 1
            if self.count_reach > 10:
                done = True
        else:
            if 'distance' in self.reward_type:
                distance_delt = self.distance_last - distance
                self.distance_last = distance
                reward += 10 * distance_delt
        msgs = self.unrealcv.read_message()
        if len(msgs) > 0:
            done = True
            reward = -10
            # print ('Collision')
        # Get observation
        '''
        arm_pose = self.unrealcv.get_arm_pose('new')
        if self.get_distance(arm_pose, self.arm_pose_last, False, 4)==0: # special case for collision
            done = True
            reward = -10
            print ('collision')
        self.arm_pose_last = arm_pose
        '''
        state = self.unrealcv.get_observation(self.cam_id, self.observation_type, self.goal_pos, action)
        info['Done'] = done
        info['Reward'] = reward
        # info['ArmPose'] = arm_pose
        return state, reward, done, info

    def _seed(self, seed=None):
        pass

    def _reset(self):
        self.launch_env()
        self.unrealcv.set_arm_pose([random.uniform(-90, 90),
                                    random.uniform(-45, 45),
                                    random.uniform(-45, 45),
                                    random.uniform(-15, 15),
                                    0], 'new')
        tip_pose = self.unrealcv.get_tip_pose()
        tip_pos_tf = self.xyz2trz(tip_pose)
        self.goal_pos = self.sample_goal()
        state = self.unrealcv.get_observation(self.cam_id, self.observation_type, self.goal_pos)

        self.count_steps = 0
        self.count_reach = 0
        self.unrealcv.set_obj_location(self.objects[0], [0, 0, -50])
        self.unrealcv.set_obj_rotation(self.objects[0], [0, 0, 0])
        self.distance_last = self.get_distance(self.goal_pos, tip_pos_tf, True)
        self.arm_pose_last = self.unrealcv.get_arm_pose('new')
        self.unrealcv.empty_msgs_buffer()

        return state

    def _close(self):
        self.unreal.close()

    def _render(self, mode='rgb_array', close=False):
        if close==True:
            self.unreal.close()
        return self.unrealcv.img_color

    def _get_action_size(self):
        return len(self.action)

    def get_distance(self, target, current, norm=False, n=3):
        error = np.array(target) - np.array(current)
        if norm:
            error = error/np.array(self.goal_range['high'])
        distance = np.linalg.norm(error[:n])
        return distance

    def xyz2trz(self, xyz):
        theta = np.arctan2(xyz[0], xyz[1])/np.pi*180
        r = np.linalg.norm(xyz[:2])
        z = xyz[2]
        return np.array([theta, r, z])

    def sample_goal(self):

        theta = random.uniform(self.goal_range['low'][0], self.goal_range['high'][0])
        r = random.uniform(self.goal_range['low'][1], self.goal_range['high'][1])
        z = random.uniform(self.goal_range['low'][2], min(r, self.goal_range['high'][2]))
        return np.array([theta, r, z])

    def load_env_setting(self, filename):
        f = open(self.get_settingpath(filename))
        type = os.path.splitext(filename)[1]
        if type == '.json':
            import json
            setting = json.load(f)
        else:
            print ('unknown type')
        return setting

    def get_settingpath(self, filename):
        import gym_unrealcv
        gympath = os.path.dirname(gym_unrealcv.__file__)
        return os.path.join(gympath, 'envs/setting', filename)
