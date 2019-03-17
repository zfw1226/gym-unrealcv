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

class UnrealCvMC(gym.Env):
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
        self.discrete_actions_player = setting['discrete_actions_player']
        self.continous_actions = setting['continous_actions']
        self.continous_actions_player = setting['continous_actions_player']
        self.max_steps = setting['max_steps']
        self.max_distance = setting['max_distance']
        self.min_distance = setting['min_distance']
        self.max_direction = setting['max_direction']
        self.max_obstacles = setting['max_obstacles']
        self.height = setting['height']
        self.pitch = setting['pitch']
        self.objects_env = setting['objects_list']
        self.reset_area = setting['reset_area']
        self.cam_area = setting['cam_area']

        self.background_list = setting['backgrounds']
        self.light_list = setting['lights']
        self.target_num = setting['target_num']  # the number of the target appearance
        self.exp_distance = setting['exp_distance']
        texture_dir = setting['imgs_dir']
        gym_path = os.path.dirname(gym_unrealcv.__file__)
        texture_dir = os.path.join(gym_path, 'envs', 'UnrealEnv', texture_dir)
        self.textures_list = os.listdir(texture_dir)
        self.safe_start = setting['safe_start']
        self.start_area = self.get_start_area(self.safe_start[0], 100)

        self.num_target = len(self.target_list)
        self.resolution = resolution
        self.num_cam = len(self.cam_id)
        self.cam_height = [setting['height'] for i in range(self.num_cam)]

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
        self.unrealcv.color_dict = self.unrealcv.build_color_dic(self.target_list)

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
        print('observation_space', self.observation_space)
        self.unrealcv.pitch = self.pitch
        # define reward type
        # distance, bbox, bbox_distance
        self.reward_type = reward_type
        self.reward_function = reward.Reward(setting)

        if self.reset_type >= 3:
            self.unrealcv.init_objects(self.objects_env)

        self.rendering = False

        self.count_close = 0

        if self.reset_type == 5:
            self.unrealcv.simulate_physics(self.objects_env)

        self.person_id = 0
        self.unrealcv.set_location(0, [self.safe_start[0][0], self.safe_start[0][1], self.safe_start[0][2]+600])
        self.unrealcv.set_rotation(0, [0, -180, -90])
        self.unrealcv.set_obj_location("TargetBP", [-3000, -3000, 220])
        if 'Random' in self.nav:
            self.random_agents = [RandomAgent(player_action_space) for i in range(self.num_target)]
        if 'Goal' in self.nav:
            self.random_agents = [GoalNavAgent(self.continous_actions_player, self.reset_area) for i in range(self.num_target)]
        if 'Internal' in self.nav:
            self.unrealcv.set_random(self.target_list[0])
            self.unrealcv.set_maxdis2goal(target=self.target_list[0], dis=500)
        if 'Interval' in self.nav:
            self.unrealcv.set_interval(30)

        self.cam_angles = np.array([0 for i in range(self.num_cam)])

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
            if 'Random' in self.nav:
                if self.action_type == 'Discrete':
                    actions2target.append(self.discrete_actions_player[self.random_agents[i].act()])
                else:
                    actions2target.append(self.random_agents[i].act())
            if 'Goal' in self.nav:
                    actions2target.append(self.random_agents[i].act(self.target_pos[i]))

        self.last_target_pos = np.array(self.target_pos).copy()

        for i, target in enumerate(self.target_list):
            self.unrealcv.set_move(target, actions2target[i][1], actions2target[i][0])

        states = []
        for i, cam in enumerate(self.cam_id):
            cam_rot = self.unrealcv.get_rotation(cam, 'hard')
            last_cam_rot = cam_rot
            cam_rot[1] += actions2cam[i][0]
            cam_rot[2] += actions2cam[i][1]
            cam_rot[2] = cam_rot[2] if cam_rot[2] < 80.0 else 80.0
            cam_rot[2] = cam_rot[2] if cam_rot[2] > - 80.0 else -80.0

            self.unrealcv.set_rotation(cam, cam_rot)
            self.cam_pose[i][-3:] = cam_rot
            self.last_cam_pose[i][-3:] = last_cam_rot
            state = self.unrealcv.get_observation(cam, self.observation_type, 'fast')
            states.append(state)
        self.states = states
            # cv2.imshow('tracker_{}'.format(str(i)), state)
        # cv2.waitKey(10)

        self.count_steps += 1

        for i, target in enumerate(self.target_list):
            self.target_pos[i] = self.unrealcv.get_obj_pose(target)

        self.cam_angles = np.array([self.last_cam_pose[i][4] for i in range(self.num_cam)])

        obj_masks = []
        bboxs = []
        directions = []
        rewards = []
        distances = []
        verti_directions = []
        cal_target_observed = np.zeros(len(self.cam_id))
        self.target_observed = np.zeros(len(self.cam_id))

        # get bbox and reward
        for i in range(len(self.cam_id)):
            # get bbox
            if self.reset_type >= 6:
                object_mask = self.unrealcv.read_image(self.cam_id[i], 'object_mask', 'fast')
                # cv2.imshow('mask of cam{}'.format(i), object_mask)
                bbox = self.unrealcv.get_bboxes(object_mask, self.target_list)
                # get bbox size
                bbox_shape = np.array(bbox[0][1]) - np.array(bbox[0][0])
                if bbox_shape[0] * bbox_shape[1] < 0.01:
                    self.target_observed[i] = 1
                bboxs.append(bbox[0])
                obj_masks.append(object_mask)

            # get relative location and reward
            direction = self.get_direction(self.cam_pose[i], self.target_pos[0])
            hori_reward = 1 - 2*abs(direction) / 45.0

            verti_direction = self.get_verti_direction(self.cam_pose[i], self.target_pos[0])
            verti_reward = 1 - 2*abs(verti_direction) / 30.0

            reward = max(hori_reward + verti_reward, -2)
            if abs(direction) <= 45.0 and abs(verti_direction) <= 40.0:
                cal_target_observed[i] = 1

            rewards.append(reward)
            directions.append(direction)
            verti_directions.append(verti_direction)
            distances.append(self.unrealcv.get_distance(self.cam_pose[i], self.target_pos[0], 3))

        info['masks'] = obj_masks
        info['bboxs'] = bboxs
        info['Ang_H'] = directions
        info['Distance'] = distances
        info['Ang_V'] = verti_directions
        info['Target_Pose'] = self.target_pos
        info['Cam_Pose'] = self.cam_pose

        # set your done condition
        if self.count_steps > self.max_steps:
            info['Done'] = True
        if sum(cal_target_observed) < 3:
            # print('observed num', sum(self.target_observed), 'count step', self.count_steps)
            self.count_close += 1
        else:
            self.count_close = 0
        if self.count_close > 20:
            info['Done'] = True

        info['Reward'] = rewards
        return self.states, info['Reward'], info['Done'], info

    def reset(self, ):
        self.C_reward = 0
        self.count_close = 0
        # stop
        for i, target in enumerate(self.target_list):
            self.unrealcv.set_move(target, 0, 0)
        np.random.seed()

        if self.reset_type >= 1:
            if self.env_name == 'MCMTRoom':
                map_id = [0, 2, 3, 7, 8, 9]
                # map_id = [1, 2, 3, 4]
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
            # self.unrealcv.set_appearance(self.target_list[1], tracker_app, spline)

        # light
        if self.reset_type >= 2:
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


        # target appearance
        if self.reset_type >= 3:
            self.unrealcv.random_player_texture(self.target_list[0], self.textures_list, 3)
            self.unrealcv.random_texture(self.background_list, self.textures_list, 3)

        # texture
        if self.reset_type >= 4:
            self.obstacles_num = np.random.randint(1, self.max_obstacles)
            self.unrealcv.clean_obstacles()
            self.unrealcv.random_obstacles(self.objects_env, self.textures_list,
                                           self.obstacles_num, self.reset_area, self.start_area)

        self.target_pos = []
        for i, target in enumerate(self.target_list):
            self.unrealcv.set_obj_location(target, self.safe_start[i])
            self.target_pos.append(self.unrealcv.get_obj_pose(target))

        # get state
        # random camera
        states = []
        self.cam_pose = []
        if self.reset_type >= 5:
            for i, cam in enumerate(self.cam_id):

                cam_loc = [np.random.randint(self.cam_area[i][0], self.cam_area[i][1]),
                           np.random.randint(self.cam_area[i][2], self.cam_area[i][3]),
                           np.random.randint(self.cam_area[i][4], self.cam_area[i][5])]
                self.unrealcv.set_location(cam, cam_loc)
                cam_rot = self.unrealcv.get_rotation(cam, 'hard')
                angle_h = self.get_direction(cam_loc+cam_rot, self.target_pos[0])
                angle_v = self.get_verti_direction(cam_loc+cam_rot, self.target_pos[0])
                cam_rot[1] += angle_h
                cam_rot[2] -= angle_v
                self.unrealcv.set_rotation(cam, cam_rot)
                self.cam_pose.append(cam_loc + cam_rot)
                states.append(self.unrealcv.get_observation(cam, self.observation_type, 'fast'))
        else:
            for i, cam in enumerate(self.cam_id):
                if i % 4 <= 1:
                    cam_loc = [self.reset_area[i % 4], (self.reset_area[2]+self.reset_area[3])/2, self.cam_height[i]]
                    cam_rot = [0, 180*(i % 4), 0]
                else:
                    cam_loc = [(self.reset_area[0]+self.reset_area[1])/2, self.reset_area[i % 4], self.cam_height[i]]
                    cam_rot = [0, 180 * (i % 4 - 2)+90, 0]

                self.unrealcv.set_location(cam, cam_loc)
                self.unrealcv.set_rotation(cam, cam_rot)
                self.cam_pose.append(cam_loc+cam_rot)
                states.append(self.unrealcv.get_observation(cam, self.observation_type, 'fast'))

        self.last_cam_pose = np.array(self.cam_pose).copy()
        self.last_target_pos = np.array(self.target_pos).copy()

        self.cam_angles = np.array([self.last_cam_pose[i][4] for i in range(self.num_cam)])

        self.count_steps = 0
        if 'Random' in self.nav or 'Goal' in self.nav:
            for i in range(len(self.random_agents)):
                self.random_agents[i].reset()
        if 'Internal' in self.nav:
            self.unrealcv.set_speed(self.target_list[0], np.random.randint(30, 200))
        self.states = states
        return self.states

    def close(self):
        self.unreal.close()

    def render(self, mode='rgb_array', close=False):
        if close==True:
            self.unreal.close()
        for i in range(self.num_cam):
            cv2.imshow('tracker_{}'.format(str(i)), self.states[i])
        cv2.waitKey(10)
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

    def get_angle(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        angle_now = np.arctan2(y_delt, x_delt) / np.pi * 180
        return angle_now

    def get_distance(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        d = np.sqrt(y_delt * y_delt + x_delt * x_delt)
        return d

    def get_verti_direction(self, current_pose, target_pose):
        # person_mid_height = target_pose[2] / 2
        person_height = target_pose[2]
        plane_distance = self.get_distance(current_pose, target_pose)
        height = current_pose[2] - person_height
        angle = np.arctan2(height, plane_distance) / np.pi * 180
        angle_now = angle + current_pose[-1]
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
            print('unknown type')

        return setting

    def get_settingpath(self, filename):
        import gym_unrealcv
        gympath = os.path.dirname(gym_unrealcv.__file__)
        return os.path.join(gympath, 'envs/setting', filename)

    def get_start_area(self, safe_start, safe_range):
        start_area = [safe_start[0] - safe_range, safe_start[0] + safe_range,
                      safe_start[1] - safe_range, safe_start[1] + safe_range]
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
    def __init__(self, action_space, goal_area, nav='New'):
        self.step_counter = 0
        self.keep_steps = 0
        self.velocity_high = action_space['high'][0]
        self.velocity_low = action_space['low'][0]
        self.angle_high = action_space['high'][1]
        self.angle_low = action_space['low'][1]
        self.goal_area = goal_area
        self.goal = self.generate_goal(self.goal_area)
        if 'Base' in nav:
            self.discrete = True
        else:
            self.discrete = False
        if 'Old' in nav:
            self.max_len = 30
        else:
            self.max_len = 100

    def act(self, pose):
        self.step_counter += 1
        if self.pose_last == None:
            self.pose_last = pose
            d_moved = 100
        else:
            d_moved = np.linalg.norm(np.array(self.pose_last) - np.array(pose))
            self.pose_last = pose
        if self.check_reach(self.goal, pose) or d_moved < 10 or self.step_counter > self.max_len:
            self.goal = self.generate_goal(self.goal_area)
            if self.discrete:
                self.velocity = (self.velocity_high + self.velocity_low)/2
            else:
                self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
            # self.velocity = 70
            self.step_counter = 0

        delt_yaw = self.get_direction(pose, self.goal)
        if self.discrete:
            if abs(delt_yaw) > self.angle_high:
                velocity = 0
            else:
                velocity = self.velocity
            if delt_yaw > 3:
                self.angle = self.angle_high / 2
            elif delt_yaw <-3:
                self.angle = self.angle_low / 2
        else:
            self.angle = np.clip(delt_yaw, self.angle_low, self.angle_high)
            # self.angle = delt_yaw
            velocity = self.velocity * (1 + 0.2*np.random.random())
        return (velocity, self.angle)

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
