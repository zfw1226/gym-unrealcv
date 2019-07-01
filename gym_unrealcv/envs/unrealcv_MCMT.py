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

class UnrealCvMCMT(gym.Env):
    def __init__(self,
                 setting_file,
                 reset_type,
                 action_type='discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 reward_type='distance',  # distance
                 docker=False,
                 resolution=(320, 240),
                 target='Ram',  # Random, Goal, Internal
                 ):
        self.docker = docker
        self.reset_type = reset_type
        self.roll = 0
        self.target = target
        setting = self.load_env_setting(setting_file)
        self.env_name = setting['env_name']
        self.cam_id = setting['cam_id']
        self.target_list = setting['targets']
        self.discrete_actions = setting['discrete_actions']
        self.discrete_actions_player = setting['discrete_actions_player']
        self.continous_actions = setting['continous_actions']
        self.continous_actions_player = setting['continous_actions_player']
        self.max_distance = setting['max_distance']
        self.min_distance = setting['min_distance']
        self.max_direction = setting['max_direction']
        self.height = setting['height']
        self.pitch = setting['pitch']
        self.objects_env = setting['objects_list']
        self.reset_area = setting['reset_area']
        self.background_list = setting['backgrounds']
        self.light_list = setting['lights']
        self.target_num = setting['target_num'] # the number of the target appearance
        self.exp_distance = setting['exp_distance']
        texture_dir = setting['imgs_dir']
        gym_path = os.path.dirname(gym_unrealcv.__file__)
        texture_dir = os.path.join(gym_path, 'envs', 'UnrealEnv', texture_dir)
        self.textures_list = os.listdir(texture_dir)
        self.safe_start = setting['safe_start']

        self.num_target = len(self.target_list)
        self.num_cam = len(self.cam_id)

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
            self.action_space = [spaces.Discrete(len(self.discrete_actions)) for i in range(self.num_cam)]
            player_action_space = spaces.Discrete(len(self.discrete_actions_player))
        elif self.action_type == 'Continuous':
            self.action_space = [spaces.Box(low=np.array(self.continous_actions['low']),
                                      high=np.array(self.continous_actions['high'])) for i in range(self.num_cam)]
            player_action_space = spaces.Discrete(len(self.continous_actions_player))

        self.count_steps = 0

        # define observation space,
        # color, depth, rgbd,...
        self.observation_type = observation_type
        assert self.observation_type in ['Color', 'Depth', 'Rgbd', 'Gray']
        self.observation_space = [self.unrealcv.define_observation(self.cam_id[i], self.observation_type, 'fast')
                                  for i in range(self.num_cam)]
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
        self.unrealcv.set_location(0, [self.safe_start[0][0], self.safe_start[0][1], self.safe_start[0][2]+600])
        self.unrealcv.set_rotation(0, [0, -180, -90])
        if 'Ram' in self.target:
            self.random_agents = [RandomAgent(player_action_space) for i in range(self.num_target)]
        if 'Nav' in self.target:
            self.random_agents = [GoalNavAgent(self.continous_actions_player, self.reset_area) for i in range(self.num_target)]
        if 'Internal' in self.target:
            self.unrealcv.set_random(self.target_list[0])
            self.unrealcv.set_maxdis2goal(target=self.target_list[0], dis=500)
        if 'Interval' in self.target:
            self.unrealcv.set_interval(30)

    def step(self, actions):
        info = dict(
            Done=False,
            Reward=[0 for i in range(self.num_cam)],
            Target_Pose=[],
            Cam_Pose=[],
            Steps=self.count_steps,
        )
        actions = np.squeeze(actions)
        actions2cam = []
        for i in range(self.num_cam):
            if self.action_type == 'Discrete':
                actions2cam.append(self.discrete_actions[actions[i]])  # delta_yaw, delta_pitch
            else:
                actions2cam.append(actions[i])  # delta_yaw, delta_pitch

        actions2target = []
        for i in range(len(self.target_list)):
            if 'Ram' in self.target:
                if self.action_type == 'Discrete':
                    actions2target.append(self.discrete_actions_player[self.random_agents[i].act()])
                else:
                    actions2target.append(self.random_agents[i].act())
            if 'Nav' in self.target:
                    actions2target.append(self.random_agents[i].act(self.target_pos[i]))

        for i, target in enumerate(self.target_list):
            self.unrealcv.set_move(target, actions2target[i][1], actions2target[i][0])

        states = []
        for i, cam in enumerate(self.cam_id):
            cam_rot = self.unrealcv.get_rotation(cam, 'hard')
            cam_rot[1] += actions2cam[i][0]
            cam_rot[2] += actions2cam[i][1]
            self.unrealcv.set_rotation(cam, cam_rot)
            self.cam_pose[i][-3:] = cam_rot
            states.append(self.unrealcv.get_observation(cam, self.observation_type, 'direct'))

        self.count_steps += 1

        for i, target in enumerate(self.target_list):
            self.target_pos[i] = self.unrealcv.get_obj_pose(target)
        info['Target_Pose'] = self.target_pos
        info['Cam_Pose'] = self.cam_pose
        # info['Direction'] = self.get_direction(info['Pose'], self.target_pos)
        # info['Distance'] = self.unrealcv.get_distance(self.target_pos, info['Pose'], 3)

        # info['Color'] = self.unrealcv.img_color
        # info['Depth'] = self.unrealcv.img_depth
        cv2.imshow('tracker_0', states[0])
        cv2.imshow('tracker_1', states[1])
        cv2.waitKey(10)

        # set your done condition
        if self.count_steps > 200:
            info['Done'] = True
        '''
        if info['Distance'] > self.max_distance or abs(info['Direction']) > self.max_direction:
            self.count_close += 1
        else:
            self.count_close = 0

        if self.count_close > 10:
           info['Done'] = True
        '''

        # add your reward function

        return states, info['Reward'], info['Done'], info

    def reset(self, ):
        self.C_reward = 0
        self.count_close = 0
        # stop
        for i, target in enumerate(self.target_list):
            self.unrealcv.set_move(target, 0, 0)
        np.random.seed()

        if self.reset_type >= 1:
            if self.env_name == 'MCMTRoom':
                #  map_id = [0, 2, 3, 7, 8, 9]
                map_id = [1, 2, 3, 4]
                spline = False
                object_app = np.random.choice(map_id)
                tracker_app = np.random.choice(map_id)
            else:
                map_id = [1, 2, 3, 4]
                spline = True
                object_app = map_id[self.person_id % len(map_id)]
                tracker_app = map_id[(self.person_id+1) % len(map_id)]
                self.person_id += 1
                # map_id = [6, 7, 8, 9]
            self.unrealcv.set_appearance(self.target_list[0], object_app, spline)
            self.unrealcv.set_appearance(self.target_list[1], tracker_app, spline)

        # target appearance
        if self.reset_type >= 2:
            if self.env_name == 'MPRoom':  # random target texture
                self.unrealcv.random_player_texture(self.target_list[0], self.textures_list, 3)
                self.unrealcv.random_player_texture(self.target_list[1], self.textures_list, 3)

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
            self.unrealcv.random_texture(self.background_list, self.textures_list, 3)

        self.target_pos = []
        for i, target in enumerate(self.target_list):
            self.unrealcv.set_obj_location(target, self.safe_start[i])
            self.target_pos.append(self.unrealcv.get_obj_pose(target))

        # get state
        states = []
        self.cam_pose = []
        for i, cam in enumerate(self.cam_id):
            if i % 4 <= 1:
                cam_loc = [self.reset_area[i % 4], (self.reset_area[2]+self.reset_area[3])/2, 300]
                cam_rot = [0, 180*(i % 4), 0]
            else:
                cam_loc = [(self.reset_area[0]+self.reset_area[1])/2, self.reset_area[i % 4], 300]
                cam_rot = [0, 90 * (i % 4), 0]  # not this

            self.unrealcv.set_location(cam, cam_loc)
            self.unrealcv.set_rotation(cam, cam_rot)
            self.cam_pose.append(cam_loc+cam_rot)
            states.append(self.unrealcv.get_observation(cam, self.observation_type, 'fast'))

        self.count_steps = 0
        if 'Ram' in self.target or 'Nav' in self.target:
            for i in range(len(self.random_agents)):
                self.random_agents[i].reset()
        if 'Internal' in self.target:
            self.unrealcv.set_speed(self.target_list[0], np.random.randint(30, 200))

        return states

    def close(self):
        self.unreal.close()

    def render(self, mode='rgb_array', close=False):
        if close==True:
            self.unreal.close()
        return self.unrealcv.img_color

    def seed(self, seed=None):
        self.person_id = seed

    def get_action_size(self):
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

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.step_counter = 0
        self.keep_steps = 0
        self.action_space = action_space

    def act(self):
        self.step_counter += 1
        if self.step_counter > self.keep_steps:
            self.action = self.action_space.sample()
            self.keep_steps = np.random.randint(1, 10)
        return self.action

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0

class GoalNavAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space, goal_area):
        self.step_counter = 0
        self.keep_steps = 0
        self.velocity_high = action_space['high'][0]
        self.velocity_low = 50
        self.angle_high = action_space['high'][1]
        self.angle_low = action_space['low'][1]
        self.goal_area = goal_area
        self.goal = self.generate_goal(self.goal_area)

    def act(self, pose):
        self.step_counter += 1
        if self.check_reach(self.goal, pose) or self.step_counter > 30:
            self.goal = self.generate_goal(self.goal_area)
            self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
            # self.velocity = 50
            self.step_counter = 0

        delt_yaw = self.get_direction(pose, self.goal)
        self.angle = np.clip(delt_yaw, self.angle_low, self.angle_high)
        return (self.velocity, self.angle)

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0
        self.goal = self.generate_goal(self.goal_area)
        self.velocity = np.random.randint(self.velocity_low, self.velocity_high)

    def generate_goal(self, goal_area):
        x = np.random.randint(goal_area[0], goal_area[1])
        y = np.random.randint(goal_area[2], goal_area[3])
        goal = np.array([x, y])
        return goal

    def check_reach(self, goal, now):
        error = np.array(now[:2]) - np.array(goal[:2])
        distance = np.linalg.norm(error)
        return distance < 50

    def get_direction(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        angle_now = np.arctan2(y_delt, x_delt) / np.pi * 180 - current_pose[4]
        if angle_now > 180:
            angle_now -= 360
        if angle_now < -180:
            angle_now += 360
        return angle_now