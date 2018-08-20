import math
import os
import time
import gym
import numpy as np
from gym import spaces
from gym_unrealcv.envs.tracking import reward
from gym_unrealcv.envs.tracking.visualization import show_info
from gym_unrealcv.envs.utils import env_unreal
from gym_unrealcv.envs.tracking.interaction import Tracking
import gym_unrealcv
import cv2
''' 
It is an env for active object tracking.

State : raw color image and depth
Action:  (linear velocity ,angle velocity) 
Done : the relative distance or angle to target is larger than the threshold.
Task: Learn to follow the target object(moving person) in the scene
'''
#TODO: fix reset rotation/ check rewards

class UnrealCvTracking_multi(gym.Env):
    def __init__(self,
                 setting_file,
                 reset_type,
                 action_type='discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 reward_type='distance',  # distance
                 docker=False,
                 resolution=(160, 120)
                 ):
        self.docker = docker
        self.reset_type = reset_type
        self.roll = 0

        setting = self.load_env_setting(setting_file)
        self.env_name = setting['env_name']
        self.cam_id = setting['cam_id']
        self.target_list = setting['targets']
        self.discrete_actions = setting['discrete_actions']
        self.continous_actions = setting['continous_actions']
        self.max_distance = setting['max_distance']
        self.min_distance = setting['min_distance']
        self.max_direction = setting['max_direction']
        self.height = setting['height']
        self.pitch = setting['pitch']
        self.objects_env = setting['objects_list']
        self.reset_area = setting['reset_area']
        self.background_list = setting['backgrounds']
        self.light_list = setting['lights']
        self.target_num = setting['target_num']
        self.exp_distance = setting['exp_distance']
        texture_dir = setting['imgs_dir']
        gym_path = os.path.dirname(gym_unrealcv.__file__)
        texture_dir = os.path.join(gym_path, 'envs', 'UnrealEnv', texture_dir)
        self.textures_list = os.listdir(texture_dir)
        self.safe_start = setting['safe_start']

        for i in range(len(self.textures_list)):
            if self.docker:
                self.textures_list[i] = os.path.join('/unreal', setting['imgs_dir'], self.textures_list[i])
            else:
                self.textures_list[i] = os.path.join(texture_dir, self.textures_list[i])

        # start unreal env
        self.unreal = env_unreal.RunUnreal(ENV_BIN=setting['env_bin'])
        env_ip, env_port = self.unreal.start(docker, resolution)

        # connect UnrealCV
        self.unrealcv = Tracking(cam_id=self.cam_id[0], port=env_port, ip=env_ip,
                                 env=self.unreal.path2env, resolution=resolution)

        # define action
        self.action_type = action_type
        assert self.action_type == 'Discrete' or self.action_type == 'Continuous'
        if self.action_type == 'Discrete':
            action_space = spaces.Discrete(len(self.discrete_actions))
        elif self.action_type == 'Continuous':
            action_space = spaces.Box(low=np.array(self.continous_actions['low']),
                                           high=np.array(self.continous_actions['high']))
        # self.action_space = [action_space, action_space]
        self.action_space = action_space
        self.count_steps = 0

        # define observation space,
        # color, depth, rgbd,...
        self.observation_type = observation_type
        assert self.observation_type in ['Color', 'Depth', 'Rgbd', 'Gray']
        observation_space = self.unrealcv.define_observation(self.cam_id[0], self.observation_type, 'direct')
        # self.observation_space = [observation_space, observation_space]
        self.observation_space = observation_space
        self.unrealcv.pitch = self.pitch
        # define reward type
        # distance, bbox, bbox_distance,
        self.reward_type = reward_type
        self.reward_function = reward.Reward(setting)

        self.rendering = False

        self.count_close = 0

        if self.reset_type == 5:
            self.unrealcv.simulate_physics(self.objects_env)

        self.person_id = 0
        self.unrealcv.set_location(0, [-475, 0, 1600])
        self.unrealcv.set_rotation(0, [0, -180, -90])

    def _step(self, actions):
        info = dict(
            Collision=False,
            Done=False,
            Trigger=0.0,
            Reward=0.0,
            Action=actions,
            Pose=[],
            Trajectory=self.trajectory,
            Steps=self.count_steps,
            Direction=None,
            Distance=None,
            Color=None,
            Depth=None,
        )
        actions = np.squeeze(actions)
        if self.action_type == 'Discrete':
            (velocity0, angle0) = self.discrete_actions[actions[0]]
            (velocity1, angle1) = self.discrete_actions[actions[1]]
        else:
            (velocity0, angle0) = actions[0]
            (velocity1, angle1) = actions[1]

        info['Collision'] = False

        # info['Collision'] = self.unrealcv.move_2d(self.cam_id, angle, velocity)
        self.unrealcv.set_move(self.target_list[0], angle0, velocity0)
        self.unrealcv.set_move(self.target_list[1], angle1, velocity1)

        self.count_steps += 1

        info['Pose'] = self.unrealcv.get_obj_pose(self.target_list[1])  # tracker pose
        self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])
        info['Direction'] = self.get_direction(info['Pose'], self.target_pos)

        info['Distance'] = self.unrealcv.get_distance(self.target_pos, info['Pose'], 2)

        # update observation
        state_0 = self.unrealcv.get_observation(self.cam_id[0], self.observation_type, 'fast')
        state_1 = self.unrealcv.get_observation(self.cam_id[1], self.observation_type, 'fast')
        states = np.array([state_0, state_1])
        info['Color'] = self.unrealcv.img_color
        info['Depth'] = self.unrealcv.img_depth
        # cv2.imshow('target', state_0)
        # cv2.imshow('tracker', state_1)
        # cv2.waitKey(10)
        if info['Distance'] > self.max_distance or info['Distance'] < self.min_distance or abs(info['Direction']) > self.max_direction:
            self.count_close += 1
        else:
            self.count_close = 0

        if self.count_close > 10:
            info['Done'] = True

        if 'distance' in self.reward_type:
            reward_1 = self.reward_function.reward_distance(info['Distance'], info['Direction'])
            reward_0 = self.reward_function.reward_target(info['Distance'], info['Direction'])
            info['Reward'] = np.array([reward_0, reward_1])
        # save the trajectory
        self.trajectory.append(info['Pose'])
        info['Trajectory'] = self.trajectory
        # self.C_reward += info['Reward']
        return states, info['Reward'], info['Done'], info

    def _reset(self, ):
        self.C_reward = 0
        self.count_close = 0
        # stop move
        self.unrealcv.set_move(self.target_list[0], 0, 0)
        self.unrealcv.set_move(self.target_list[1], 0, 0)
        np.random.seed()
        #  self.exp_distance = np.random.randint(150, 250)

        # target appearance
        if self.reset_type >= 2:
            map_id = [2, 3, 6, 7, 9]
            self.unrealcv.set_appearance(self.target_list[0], np.random.choice(map_id))
            self.unrealcv.set_appearance(self.target_list[1], np.random.choice(map_id))
            #  map_id = [0, 2, 3, 7, 8, 9]
            if self.env_name == 'MPRoom':  # random target texture
                for i in range(5):
                    self.unrealcv.set_texture(self.target_list[0], (1, 1, 1), np.random.uniform(0, 1, 3),
                                              self.textures_list[np.random.randint(0, len(self.textures_list))],
                                              np.random.randint(2, 6), i)
                for i in range(5):
                    self.unrealcv.set_texture(self.target_list[1], (1, 1, 1), np.random.uniform(0, 1, 3),
                                              self.textures_list[np.random.randint(0, len(self.textures_list))],
                                              np.random.randint(2, 6), i)

        # light
        if self.reset_type >= 3:
            for lit in self.light_list:
                if 'sky' in lit:
                    self.unrealcv.set_skylight(lit, [1, 1, 1], np.random.uniform(0.5,2))
                else:
                    lit_direction = np.random.uniform(-1, 1, 3)
                    if 'directional' in lit:
                        lit_direction[0] = lit_direction[0] * 60
                        lit_direction[1] = lit_direction[1] * 80
                        lit_direction[2] = lit_direction[2] * 60
                    else:
                        lit_direction *= 180
                    self.unrealcv.set_light(lit, lit_direction, np.random.uniform(1, 4), np.random.uniform(0.1,1,3))

        # texture
        if self.reset_type >= 4:
            self.unrealcv.random_texture(self.background_list, self.textures_list)

        self.unrealcv.set_obj_location(self.target_list[0], [-600, -200, 250])
        self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])
        res = self.unrealcv.get_startpoint(self.target_pos, self.exp_distance, self.reset_area, self.height)

        count = 0
        while not res:
            count += 1
            time.sleep(0.1)
            self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])
            res = self.unrealcv.get_startpoint(self.target_pos, self.exp_distance, self.reset_area)
        cam_pos_exp, yaw = res
        cam_pos_exp[-1] = self.height
        self.unrealcv.set_obj_location(self.target_list[1], cam_pos_exp)
        yaw_pre = self.unrealcv.get_obj_rotation(self.target_list[1])[1]
        delta_yaw = yaw-yaw_pre
        self.unrealcv.set_move(self.target_list[1], delta_yaw, 0)
        # self.unrealcv.set_obj_rotation(self.target_list[1], [self.roll, yaw, self.pitch])
        time.sleep(0.5)
        current_pose = self.unrealcv.get_obj_pose(self.target_list[1])

        # get state
        state_0 = self.unrealcv.get_observation(self.cam_id[0], self.observation_type, 'fast')
        state_1 = self.unrealcv.get_observation(self.cam_id[1], self.observation_type, 'fast')
        states = np.array([state_0, state_1])
        # cv2.imshow('tracker', state_1)
        # cv2.waitKey(10)
        self.trajectory = []
        self.trajectory.append(current_pose)
        self.count_steps = 0

        return states

    def _close(self):
        self.unreal.close()

    def _render(self, mode='rgb_array', close=False):
        if close==True:
            self.unreal.close()
        return self.unrealcv.img_color

    def _seed(self, seed=None):
        self.person_id = seed

    def _get_action_size(self):
        return len(self.action)

    def get_direction(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        angle_now = np.arctan2(y_delt, x_delt)/np.pi*180-current_pose[4]
        if angle_now > 180:
            angle_now -= 360
        if angle_now < -180:
            angle_now += 360
        return angle_now

    def load_env_setting(self, filename):
        gym_path = os.path.dirname(gym_unrealcv.__file__)
        setting_path = os.path.join(gym_path, 'envs', 'setting', filename)

        f = open(setting_path)
        f_type = os.path.splitext(filename)[1]
        if f_type == '.json':
            import json
            setting = json.load(f)
        else:
            print ('unknown type')

        return setting

    def get_settingpath(self, filename):
        import gym_unrealcv
        gympath = os.path.dirname(gym_unrealcv.__file__)
        return os.path.join(gympath, 'envs/setting', filename)
