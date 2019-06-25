import time
import gym
import numpy as np
from gym import spaces
from gym_unrealcv.envs.tracking import reward
from gym_unrealcv.envs.utils import env_unreal, misc
from gym_unrealcv.envs.navigation.interaction import Navigation
import random

''' 
It is an env for active object tracking.

State : raw color image and depth
Action:  (linear velocity ,angle velocity) 
Done : the relative distance or angle to target is larger than the threshold.
Task: Learn to follow the target object(moving person) in the scene
'''


class UnrealCvTracking_spline(gym.Env):
    def __init__(self,
                 setting_file,
                 reset_type='Random',  # random
                 action_type='discrete',  # 'discrete', 'continuous'
                 observation_type='color',  # 'color', 'depth', 'rgbd'
                 reward_type='distance',  # distance
                 docker=False,
                 resolution=(80, 80)
                 ):
        setting = misc.load_env_setting(setting_file)
        self.cam_id = setting['cam_id']
        self.target_list = setting['targets']
        self.discrete_actions = setting['discrete_actions']
        self.continous_actions = setting['continous_actions']
        self.max_distance = setting['max_distance']
        self.max_direction = setting['max_direction']
        self.objects_env = setting['objects_list']

        self.docker = docker
        self.reset_type = reset_type
        self.roll = 0
        self.pitch = 0
        self.count_steps = 0

        # start unreal env
        self.unreal = env_unreal.RunUnreal(ENV_BIN=setting['env_bin'])
        env_ip, env_port = self.unreal.start(docker, resolution)

        # connect UnrealCV
        self.unrealcv = Navigation(cam_id=self.cam_id, port= env_port,
                                   ip=env_ip, env=self.unreal.path2env,
                                   resolution=resolution)

        # define action
        self.action_type = action_type
        assert self.action_type == 'Discrete' or self.action_type == 'Continuous'
        if self.action_type == 'Discrete':
            self.action_space = spaces.Discrete(len(self.discrete_actions))
        elif self.action_type == 'Continuous':
            self.action_space = spaces.Box(low=np.array(self.continous_actions['low']),
                                           high=np.array(self.continous_actions['high']))

        # define observation space,
        #  color, depth, rgbd,...
        self.observation_type = observation_type
        assert self.observation_type == 'Color' or self.observation_type == 'Depth' or self.observation_type == 'Rgbd'
        self.observation_space = self.unrealcv.define_observation(self.cam_id, self.observation_type)

        # define reward
        self.reward_function = reward.Reward(setting)

        self.rendering = False

        # init augment env
        if 'Random' in self.reset_type:
            self.show_list = self.objects_env
            self.hiden_list = random.sample(self.objects_env, min(15, len(self.objects_env)))
            for x in self.hiden_list:
                self.show_list.remove(x)
                self.unrealcv.hide_obj(x)

    def render(self, mode='rgb_array', close=False):
        if close==True:
            self.unreal.close()
        return self.unrealcv.img_color

    def step(self, action):
        info = dict(
            Collision=False,
            Done=False,
            Reward=0.0,
            Action=action,
            Pose=[],
            Trajectory=self.trajectory,
            Steps=self.count_steps,
            Direction=None,
            Distance=None,
            Color=None,
            Depth=None,
        )
        action = np.squeeze(action)
        if self.action_type == 'Discrete':
            # linear
            (velocity, angle) = self.discrete_actions[action]
        else:
            (velocity, angle) = action

        info['Collision'] = self.unrealcv.move_2d(self.cam_id, angle, velocity)

        self.count_steps += 1

        info['Pose'] = self.unrealcv.get_pose(self.cam_id)
        self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])
        info['Direction'] = misc.get_direction(info['Pose'], self.target_pos)
        info['Distance'] = self.unrealcv.get_distance(self.target_pos, info['Pose'], 2)
        info['Reward'] = self.reward_function.reward_distance(info['Distance'], info['Direction'])

        # update observation
        state = self.unrealcv.get_observation(self.cam_id, self.observation_type, 'direct')
        info['Color'] = self.unrealcv.img_color
        info['Depth'] = self.unrealcv.img_depth

        # done condition
        if info['Distance'] > self.max_distance or abs(info['Direction']) > self.max_direction:
            self.count_close += 1
        if self.count_close > 10:
            info['Done'] = True
            info['Reward'] = -1

        # save the trajectory
        self.trajectory.append(info['Pose'])
        info['Trajectory'] = self.trajectory

        self.C_reward += info['Reward']
        return state, info['Reward'], info['Done'], info

    def reset(self, ):
        self.C_reward = 0
        self.count_close = 0

        self.target_pos = self.unrealcv.get_obj_pose(self.target_list[0])
        # random hide and show objects
        if 'Random' in self.reset_type:
            num_update = 5
            objs_to_hide = random.sample(self.show_list, num_update)
            for x in objs_to_hide:
                self.show_list.remove(x)
                self.hiden_list.append(x)
                self.unrealcv.hide_obj(x)
            objs_to_show = random.sample(self.hiden_list[:-num_update], num_update)
            for x in objs_to_show:
                self.hiden_list.remove(x)
                self.show_list.append(x)
                self.unrealcv.show_obj(x)
            time.sleep(0.5 * random.random())

        time.sleep(1)
        cam_pos = self.target_pos
        self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])

        # set pose
        self.unrealcv.set_location(self.cam_id, cam_pos[:3])
        self.unrealcv.set_rotation(self.cam_id, cam_pos[-3:])
        current_pose = self.unrealcv.get_pose(self.cam_id, 'soft')

        # get state
        time.sleep(0.1)
        state = self.unrealcv.get_observation(self.cam_id, self.observation_type)

        self.trajectory = []
        self.trajectory.append(current_pose)
        self.count_steps = 0

        return state

    def close(self):
        self.unreal.close()

    def seed(self, seed=None):
        pass

    def render(self, mode='rgb_array', close=False):
        if close==True:
            self.unreal.close()
        return self.unrealcv.img_color
