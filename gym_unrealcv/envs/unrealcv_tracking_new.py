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
import random

''' 
It is an env for active object tracking.

State : raw color image and depth
Action:  (linear velocity ,angle velocity) 
Done : the relative distance or angle to target is larger than the threshold.
Task: Learn to follow the target object(moving person) in the scene
'''


class UnrealCvTracking_base_random(gym.Env):
    def __init__(self,
                 setting_file,
                 reset_type,
                 action_type='discrete',  # 'discrete', 'continuous'
                 observation_type='color', # 'color', 'depth', 'rgbd'
                 reward_type='distance', # distance
                 docker=False,
                 test=False,
                 resolution=(160, 120)
                 ):
        self.docker = docker
        self.reset_type = reset_type
        self.test = test
        self.roll = 0

        setting = self.load_env_setting(setting_file)

        # start unreal env
        self.unreal = env_unreal.RunUnreal(ENV_BIN=setting['env_bin'])
        env_ip, env_port = self.unreal.start(docker, resolution)

        # connect UnrealCV
        self.unrealcv = Tracking(cam_id=self.cam_id, port=env_port, ip=env_ip,
                                 env=self.unreal.path2env, resolution=resolution)

        # define action
        self.action_type = action_type
        assert self.action_type == 'discrete' or self.action_type == 'continuous'
        if self.action_type == 'discrete':
            self.action_space = spaces.Discrete(len(self.discrete_actions))
        elif self.action_type == 'continuous':
            self.action_space = spaces.Box(low=np.array(self.continous_actions['low']),
                                           high=np.array(self.continous_actions['high']))

        self.count_steps = 0

        # define observation space,
        # color, depth, rgbd,...
        self.observation_type = observation_type
        assert self.observation_type == 'color' or self.observation_type == 'depth' or self.observation_type == 'rgbd'
        self.observation_space = self.unrealcv.define_observation(self.cam_id,self.observation_type, 'fast')


        self.unrealcv.pitch = self.pitch
        # define reward type
        # distance, bbox, bbox_distance,
        self.reward_type = reward_type
        self.reward_function = reward.Reward(setting)

        self.rendering = False
        if self.reset_type == 0:  # spline
            self.unrealcv.client.request('vbp {target} reset'.format(target=self.target_list[0]))
        self.unrealcv.start_walking(self.target_list[0])
        self.count_close = 0

        if self.reset_type == 5:
            self.unrealcv.simulate_physics(self.objects_env)

        self.person_id = 0
        if self.test:
            self.reward_dic = dict()
            self.ep_dic = dict()
            for i in range(0,4):
                self.reward_dic[i] = []
                self.ep_dic[i] = []

    def _step(self, action ):
        info = dict(
            Collision=False,
            Done=False,
            Trigger=0.0,
            Maxstep=False,
            Reward=0.0,
            Action = action,
            Pose=[],
            Trajectory=self.trajectory,
            Steps=self.count_steps,
            Direction=None,
            Distance=None,
            Color=None,
            Depth=None,
        )
        action = np.squeeze(action)
        if self.action_type == 'discrete':
            # linear
            (velocity, angle) = self.discrete_actions[action]
        else:
            (velocity, angle) = action

        info['Collision'] = self.unrealcv.move_2d(self.cam_id, angle, velocity)

        self.count_steps += 1

        info['Pose'] = self.unrealcv.get_pose(self.cam_id)
        self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])
        info['Direction'] = self.get_direction(self.target_pos, info['Pose'][:3]) - info['Pose'][-2]
        if info['Direction'] < -180:
            info['Direction'] += 360
        elif info['Direction'] > 180:
            info['Direction'] -= 360
        info['Distance'] = self.get_distance(self.target_pos,info['Pose'][:3])

        # update observation
        state = self.unrealcv.get_observation(self.cam_id, self.observation_type, 'fast')
        info['Color'] = self.unrealcv.img_color
        info['Depth'] = self.unrealcv.img_depth

        if info['Distance'] > self.max_distance or info['Distance'] < self.min_distance or abs(info['Direction'])> self.max_direction:
            self.count_close += 1
        else:
            self.count_close = 0

        if  self.count_close > 5:
            info['Done'] = True
            info['Reward'] = -1
        elif 'distance' in self.reward_type:
            info['Reward'] = self.reward_function.reward_distance(info['Distance'],info['Direction'])

        # limit the max steps of every episode
        if self.count_steps > self.max_steps:
           info['Done'] = True
           info['Maxstep'] = True
           print ('Reach Max Steps')

        # save the trajectory
        self.trajectory.append(info['Pose'])
        info['Trajectory'] = self.trajectory

        if self.rendering:
            show_info(info,self.action_type)

        if info['Done'] and self.test:
            self.reward_dic[self.person_id].append(self.C_reward)
            self.ep_dic[self.person_id].append(self.count_steps)
            print ('reward', self.reward_dic)
            print ('ep_len', self.ep_dic)

        self.C_reward += info['Reward']
        return state, info['Reward'], info['Done'], info

    def _reset(self, ):
        self.C_reward = 0
        self.count_close = 0
        self.unrealcv.start_walking(self.target_list[0])  # stop moving
        np.random.seed()

        if self.reset_type == -1:
            self.person_id = (self.person_id + 1) % 4
            self.unrealcv.set_appearance(self.target_list[0],self.person_id+6,True)

        if self.reset_type == 0:  # spline
            self.unrealcv.client.request('vbp {target} reset'.format(target=self.target_list[0]))
            time.sleep(0.3)

        if self.reset_type >= 1:  # random walk
            self.unrealcv.set_speed(self.target_list[0],np.random.randint(50,150))
            self.unrealcv.set_acceleration(self.target_list[0], np.random.randint(100, 300))
            self.unrealcv.set_maxdis2goal(self.target_list[0], np.random.randint(500, 3000))
            if self.reset_type == 1:
                self.unrealcv.set_appearance(self.target_list[0], 7)

        if self.reset_type == 2 or self.reset_type ==3 or self.reset_type ==4 or self.reset_type == 5: # appearance
            self.unrealcv.set_appearance(self.target_list[0], np.random.randint(0, self.target_num))

        if self.reset_type == 3 or self.reset_type == 4 or self.reset_type == 5: # light
            for lit in self.light_list:
                if 'sky' in lit:
                    self.unrealcv.set_skylight(lit, [1,1,1], np.random.uniform(0.5,2))
                else:
                    lit_direction = np.random.uniform(-1, 1, 3)
                    if 'directional' in lit:
                        lit_direction[0] = lit_direction[0] * 60
                        lit_direction[1] = lit_direction[1] * 80
                        lit_direction[2] = lit_direction[2] * 60
                    else:
                        lit_direction *= 180
                    self.unrealcv.set_light(lit, lit_direction, np.random.uniform(1,4), np.random.uniform(0.1,1,3))

        if self.reset_type == 4 or self.reset_type == 5:  # texture
            self.unrealcv.random_texture(self.background_list, self.textures_list)

        if self.reset_type == 5:  # layout
            self.unrealcv.random_layout(self.objects_env, self.reset_area)

        self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])
        res = self.unrealcv.get_startpoint(self.target_pos, self.exp_distance, self.reset_area, self.height)
        while not res:
            self.unrealcv.reset_target(self.target_list[0])
            time.sleep(0.1)
            self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])
            res = self.unrealcv.get_startpoint(self.target_pos, self.exp_distance, self.reset_area)
        cam_pos_exp, yaw = res

        self.unrealcv.set_location(self.cam_id, cam_pos_exp) # moveto

        '''
        error = self.unrealcv.error_position(pose_now[:3], cam_pos_exp)
        if error>10:
           #self.target_pos = random.sample(self.safe_start,1)[0]
           #self.unrealcv.set_object_location(self.target_list[0],self.target_pos) # [-600, 400 ,100]
           #self.unrealcv.reset_target(self.target_list[0])
           #self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])
           cam_pos_exp, yaw = self.unrealcv.get_startpoint(self.target_pos,self.exp_distance, self.reset_area)
           self.unrealcv.set_location(self.cam_id, cam_pos_exp)
        '''

        self.unrealcv.set_rotation(self.cam_id,[self.roll, yaw, self.pitch])
        current_pose = self.unrealcv.get_pose(self.cam_id, 'soft')

        # get state
        time.sleep(0.5)
        state = self.unrealcv.get_observation(self.cam_id, self.observation_type, 'fast')

        self.trajectory = []
        self.trajectory.append(current_pose)
        self.count_steps = 0
        while True:
            if self.unrealcv.start_walking(self.target_list[0]):  # stop moving
                break

        return state

    def _close(self):
        if self.docker:
            self.unreal.docker.close()

    def _seed(self, seed=None):
        print('fake seed')

    def _get_action_size(self):
        return len(self.action)

    def get_distance(self, target, current):
        error = abs(np.array(target)[:2] - np.array(current)[:2])# only x and y
        distance = math.sqrt(sum(error * error))
        return distance

    def get_direction(self, target_pos, camera_pos):
        relative_pos = np.array(target_pos) - np.array(camera_pos)
        if relative_pos[0] > 0:
            direction = 180 * np.arctan(relative_pos[1] / relative_pos[0]) / np.pi
        else:
            direction = 180 + 180 * np.arctan(relative_pos[1] / min(relative_pos[0], -0.0001)) / np.pi

        return direction

    def load_env_setting(self, filename):
        import gym_unrealcv
        gym_path = os.path.dirname(gym_unrealcv.__file__)
        setting_path = os.path.join(gym_path, 'envs', 'setting', filename)

        f = open(setting_path)
        f_type = os.path.splitext(filename)[1]
        if f_type == '.json':
            import json
            setting = json.load(f)
        else:
            print ('unknown type')

        self.cam_id = setting['cam_id']
        self.target_list = setting['targets']
        self.max_steps = setting['max_steps']
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
        texture_dir = os.path.join(gym_path, 'envs', 'UnrealEnv', texture_dir)
        self.textures_list = os.listdir(texture_dir)
        self.safe_start = setting['safe_start']

        for i in range(len(self.textures_list)):
            if self.docker:
                self.textures_list[i] = os.path.join('/unreal', setting['imgs_dir'], self.textures_list[i])
            else:
                self.textures_list[i] = os.path.join(texture_dir, self.textures_list[i])

        return setting

    def get_settingpath(self, filename):
        import gym_unrealcv
        gympath = os.path.dirname(gym_unrealcv.__file__)
        return os.path.join(gympath, 'envs/setting', filename)
