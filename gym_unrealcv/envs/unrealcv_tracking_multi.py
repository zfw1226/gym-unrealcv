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
                 resolution=(160, 120),
                 nav='Random',  # Random, Goal, Internal
                 ):
        self.docker = docker
        self.reset_type = reset_type
        self.roll = 0
        self.nav = nav
        setting = self.load_env_setting(setting_file)
        self.env_name = setting['env_name']
        self.cam_id = setting['cam_id']
        self.target_list = setting['targets']
        self.discrete_actions = setting['discrete_actions']
        self.continous_actions = setting['continous_actions']
        self.continous_actions_forward = setting['continous_actions_forward']
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
        self.start_area = self.get_start_area(self.safe_start[0], 500)

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
            action_space_forward = action_space
        elif self.action_type == 'Continuous':
            action_space = spaces.Box(low=np.array(self.continous_actions['low']),
                                      high=np.array(self.continous_actions['high']))
            action_space_forward = spaces.Box(low=np.array(self.continous_actions_forward['low']),
                                              high=np.array(self.continous_actions_forward['high']))
        # self.action_space = [action_space, action_space]
        self.action_space = [action_space_forward, action_space]

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

        if self.reset_type >= 5:
            self.unrealcv.init_objects(self.objects_env)
            # self.unrealcv.simulate_physics(self.objects_env)

        self.person_id = 0
        self.count_eps = 0
        self.count_steps = 0
        self.count_close = 0
        self.direction = None
        self.rendering = False
        # set third-person view camera
        self.unrealcv.set_location(0, [self.safe_start[0][0], self.safe_start[0][1], self.safe_start[0][2]+600])
        self.unrealcv.set_rotation(0, [0, -180, -90])
        # config target
        if 'Random' in self.nav:
            self.random_agent = RandomAgent(action_space_forward)
        if 'Goal' in self.nav:
            self.random_agent = GoalNavAgent(self.continous_actions_forward, self.reset_area)
        if 'Internal' in self.nav:
            self.unrealcv.set_random(self.target_list[0])
            self.unrealcv.set_maxdis2goal(target=self.target_list[0], dis=500)
        if 'Interval' in self.nav:
            self.unrealcv.set_interval(setting['interval'])
        self.w_p = 1.0
        self.ep_lens = []

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

        if 'Random' in self.nav:
            if self.action_type == 'Discrete':
                (velocity0, angle0) = self.discrete_actions[self.random_agent.act()]
            else:
                (velocity0, angle0) = self.random_agent.act()
        if 'Goal' in self.nav:
            (velocity0, angle0) = self.random_agent.act(self.target_pos)

        # info['Collision'] = self.unrealcv.get_hit(self.target_list[1])

        self.unrealcv.set_move(self.target_list[0], angle0, velocity0)
        self.unrealcv.set_move(self.target_list[1], angle1, velocity1)

        self.count_steps += 1

        info['Pose'] = self.unrealcv.get_obj_pose(self.target_list[1])  # tracker pose
        self.target_pos = self.unrealcv.get_obj_pose(self.target_list[0])
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

        if 'distance' in self.reward_type:
            reward_1 = self.reward_function.reward_distance(info['Distance'], info['Direction'])
            if reward_1 > -1:
                reward_0 = - reward_1
            else:
                reward_0 = self.reward_function.reward_target(info['Distance'], info['Direction'], None, self.w_p)
            info['Reward'] = np.array([reward_0, reward_1])

        if reward_1 <= -0.99:
            self.count_close += 1
        else:
            self.count_close = 0

        if self.count_close > 20:
           info['Done'] = True
        # save the trajectory
        self.trajectory.append([info['Distance'], info['Direction']])
        info['Trajectory'] = self.trajectory

        return states, info['Reward'], info['Done'], info

    def _reset(self, ):
        self.C_reward = 0
        self.count_close = 0
        self.count_eps += 1
        self.ep_lens.append(self.count_steps)

        # adaptive weight
        if 'Dynamic' in self.nav:
            ep_lens_mean = np.array(self.ep_lens[-100:]).mean()
            self.w_p = 1 - int(ep_lens_mean/100)/5.0
        else:
            self.w_p = 0
        self.count_steps = 0
        # stop move
        self.unrealcv.set_move(self.target_list[0], 0, 0)
        self.unrealcv.set_move(self.target_list[1], 0, 0)
        np.random.seed()
        #  self.exp_distance = np.random.randint(150, 250)
        self.unrealcv.set_obj_location(self.target_list[0], self.safe_start[0])
        if self.reset_type >= 1:
            if self.env_name == 'MPRoom':
                #  map_id = [0, 2, 3, 7, 8, 9]
                map_id = [2, 3, 6, 7, 9]
                spline = False
                object_app = np.random.choice(map_id)
                tracker_app = np.random.choice(map_id)
            else:
                map_id = [1, 2, 3, 4]
                spline = True
                object_app = map_id[self.person_id % len(map_id)]
                tracker_app = map_id[self.person_id % len(map_id)]
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
                    self.unrealcv.set_skylight(lit, [1, 1, 1], np.random.uniform(0.5, 2))
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

        # obstacle
        if self.reset_type >= 5:
            self.unrealcv.clean_obstacles()
            self.unrealcv.random_obstacles(self.objects_env, self.textures_list,
                                           20, self.reset_area, self.start_area)


        self.target_pos = self.unrealcv.get_obj_pose(self.target_list[0])

        res = self.unrealcv.get_startpoint(self.target_pos, self.exp_distance, self.reset_area, self.height, self.direction)

        count = 0
        while not res:
            count += 1
            time.sleep(0.1)
            self.target_pos = self.unrealcv.get_obj_pose(self.target_list[0])
            res = self.unrealcv.get_startpoint(self.target_pos, self.exp_distance, self.reset_area)
        cam_pos_exp, yaw = res
        cam_pos_exp[-1] = self.height
        self.unrealcv.set_obj_location(self.target_list[1], cam_pos_exp)
        yaw_pre = self.unrealcv.get_obj_rotation(self.target_list[1])[1]
        delta_yaw = yaw-yaw_pre
        while abs(delta_yaw) > 3:
            self.unrealcv.set_move(self.target_list[1], delta_yaw, 0)
            yaw_pre = self.unrealcv.get_obj_rotation(self.target_list[1])[1]
            delta_yaw = (yaw - yaw_pre) % 360
            if delta_yaw > 180:
                delta_yaw = 360 - delta_yaw

        current_pose = self.unrealcv.get_obj_pose(self.target_list[1])
        # get state
        state_0 = self.unrealcv.get_observation(self.cam_id[0], self.observation_type, 'fast')
        state_1 = self.unrealcv.get_observation(self.cam_id[1], self.observation_type, 'fast')
        states = np.array([state_0, state_1])
        # cv2.imshow('tracker', state_1)
        # cv2.waitKey(10)
        self.trajectory = []
        self.trajectory.append(current_pose)
        if 'Random' in self.nav or 'Goal' in self.nav:
            self.random_agent.reset()
        if 'Internal' in self.nav:
            self.unrealcv.set_speed(self.target_list[0], np.random.randint(30, 200))

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

    def get_start_area(self, safe_start, safe_range):
        start_area = [safe_start[0]-safe_range, safe_start[0]+safe_range,
                     safe_start[1]-safe_range, safe_start[1]+safe_range]
        return start_area

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
        if self.pose_last == None:
            self.pose_last = pose
            d_moved = 100
        else:
            d_moved = np.linalg.norm(np.array(self.pose_last) - np.array(pose))
            self.pose_last = pose
        if self.check_reach(self.goal, pose) or d_moved < 5:
            self.goal = self.generate_goal(self.goal_area)
            self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
            # self.velocity = 70
            self.step_counter = 0

        delt_yaw = self.get_direction(pose, self.goal)
        self.angle = np.clip(delt_yaw, self.angle_low, self.angle_high)
        return (self.velocity, self.angle)

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0
        self.goal = self.generate_goal(self.goal_area)
        self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
        self.pose_last = None

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