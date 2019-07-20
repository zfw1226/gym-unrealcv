import os
import time
import gym
import numpy as np
from gym import spaces
from gym_unrealcv.envs.tracking import reward, baseline
from gym_unrealcv.envs.utils import env_unreal, misc
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
                 resolution=(160, 160),
                 target='Ram',  # Random, Goal, Internal
                 ):
        self.docker = docker
        self.reset_type = reset_type
        self.roll = 0
        self.target = target
        setting = misc.load_env_setting(setting_file)
        self.env_name = setting['env_name']
        self.cam_id_all = setting['cam_id']
        self.player_list = setting['players']
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
        self.max_player_num = setting['max_player_num']  # the max players number
        self.exp_distance = setting['exp_distance']
        texture_dir = setting['imgs_dir']
        gym_path = os.path.dirname(gym_unrealcv.__file__)
        texture_dir = os.path.join(gym_path, 'envs', 'UnrealEnv', texture_dir)
        self.textures_list = os.listdir(texture_dir)
        self.safe_start = setting['safe_start']
        self.start_area = self.get_start_area(self.safe_start[0], 100)

        self.num_target = len(self.player_list)
        self.resolution = resolution
        self.num_cam = len(self.cam_id_all)
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
        self.unrealcv = Tracking(cam_id=self.cam_id_all[0], port=env_port, ip=env_ip,
                                 env=self.unreal.path2env, resolution=resolution)
        self.unrealcv.color_dict = self.unrealcv.build_color_dic(self.player_list)

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
        self.observation_space = [self.unrealcv.define_observation(self.cam_id_all[i], self.observation_type, 'fast')
                                  for i in range(self.num_cam)]

        self.unrealcv.pitch = self.pitch
        # define reward type
        # distance, bbox, bbox_distance
        self.reward_type = reward_type
        self.reward_function = reward.Reward(setting)

        if self.reset_type >= 3:
            self.unrealcv.init_objects(self.objects_env)

        self.rendering = False

        self.count_close = 0
        self.person_id = 0
        self.unrealcv.set_location(0, [self.safe_start[0][0], self.safe_start[0][1], self.safe_start[0][2]+600])
        self.unrealcv.set_rotation(0, [0, -180, -90])

        if 'Ram' in self.target:
            self.random_agents = [baseline.RandomAgent(player_action_space) for i in range(self.max_player_num)]
        elif 'Nav' in self.target:
            self.random_agents = [baseline.GoalNavAgent(self.continous_actions_player, self.reset_area, self.target)
                                  for i in range(self.max_player_num)]

        self.unrealcv.set_interval(50)

    def step(self, actions):
        info = dict(
            Done=False,
            Reward=[0 for i in range(self.num_cam)],
            Target_Pose=[],
            Cam_Pose=[],
        )
        self.count_steps += 1
        actions = np.squeeze(actions)
        actions2cam = []
        for i in range(self.num_cam):
            if self.action_type == 'Discrete':
                actions2cam.append(self.discrete_actions[actions[i]])  # delta_yaw, delta_pitch
            else:
                actions2cam.append(actions[i])  # delta_yaw, delta_pitch

        actions2target = []
        for i in range(len(self.player_list)):
            if 'Ram' not in self.target and 'Nav' not in self.target:
                if self.action_type == 'Discrete':
                    actions2target.append(self.discrete_actions[actions[i]])
                else:
                    actions2target.append(actions[i])
            else:
                if 'Ram' in self.target:
                    if self.action_type == 'Discrete':
                        actions2target.append(self.discrete_actions_player[self.random_agents[i].act(self.obj_pos[i])])
                    else:
                        actions2target.append(self.random_agents[i].act(self.obj_pos[i]))
                if 'Nav' in self.target:
                    actions2target.append(self.random_agents[i].act(self.obj_pos[i]))

        # send control commands to env
        self.cam_rots = []
        for i, cam in enumerate(self.cam_id):
            cam_rot = self.cam_pose[i][-3:]
            cam_rot[1] += actions2cam[i][0]
            cam_rot[2] += actions2cam[i][1]
            cam_rot[2] = cam_rot[2] if cam_rot[2] < 80.0 else 80.0
            cam_rot[2] = cam_rot[2] if cam_rot[2] > - 80.0 else -80.0
            self.cam_rots.append(cam_rot)
        self.unrealcv.set_move_with_cam_batch(self.player_list, actions2target, self.cam_id, self.cam_rots)
        # get observation
        self.states, self.obj_pos, self.cam_rots = self.unrealcv.get_pose_img_batch(self.player_list, self.cam_id,
                                                                                    'lit', 'bmp', True)
        for i, cam_rot in enumerate(self.cam_rots):
            self.cam_pose[i][-3:] = cam_rot
        self.unrealcv.img_color = self.states[0]

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
                bbox = self.unrealcv.get_bboxes(object_mask, self.player_list)
                # get bbox size
                bbox_shape = np.array(bbox[0][1]) - np.array(bbox[0][0])
                if bbox_shape[0] * bbox_shape[1] < 0.01:
                    self.target_observed[i] = 1
                bboxs.append(bbox)

            # get relative location and reward
            direction = self.get_direction(self.cam_pose[i], self.obj_pos[0])
            hori_reward = 1 - 2*abs(direction) / 45.0

            verti_direction = self.get_verti_direction(self.cam_pose[i], self.obj_pos[0])
            verti_reward = 1 - 2*abs(verti_direction) / 30.0

            reward = max(hori_reward + verti_reward, -2)
            if abs(direction) <= 45.0 and abs(verti_direction) <= 40.0:
                cal_target_observed[i] = 1

            rewards.append(reward)
            directions.append(direction)
            verti_directions.append(verti_direction)
            distances.append(self.unrealcv.get_distance(self.cam_pose[i], self.obj_pos[0], 3))

        info['bboxs'] = bboxs
        info['Ang_H'] = directions
        info['Distance'] = distances
        info['Ang_V'] = verti_directions
        info['Target_Pose'] = self.obj_pos
        info['Cam_Pose'] = self.cam_pose

        # set your done condition
        if self.count_steps > self.max_steps:
            info['Done'] = True
        if sum(cal_target_observed) < 3:
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
        for i, target in enumerate(self.player_list):
            self.unrealcv.set_move(target, 0, 0)
        np.random.seed()
        self.cam_id = self.cam_id_all[:self.num_cam]
        if self.reset_type >= 1:
            for obj in self.player_list:
                if self.env_name == 'MPRoom':
                    #  map_id = [0, 2, 3, 7, 8, 9]
                    map_id = [2, 3, 6, 7, 9]
                    spline = False
                    app_id = np.random.choice(map_id)
                else:
                    map_id = [1, 2, 3, 4]
                    spline = True
                    app_id = map_id[self.person_id % len(map_id)]
                    self.person_id += 1
                    # map_id = [6, 7, 8, 9]
                self.unrealcv.set_appearance(obj, app_id, spline)

        # light
        if self.reset_type >= 2:
            self.unrealcv.random_lit(self.light_list)

        # texture
        if self.reset_type >= 3:
            for _, target in enumerate(self.player_list):
                self.unrealcv.random_player_texture(target, self.textures_list, 3)
            self.unrealcv.random_texture(self.background_list, self.textures_list, -1)

        if self.reset_type >= 4:
            self.obstacles_num = self.max_obstacles
            self.unrealcv.clean_obstacles()
            self.unrealcv.random_obstacles(self.objects_env, self.textures_list,
                                           self.obstacles_num, self.reset_area, self.start_area)

        for i, target in enumerate(self.player_list):
            self.unrealcv.set_obj_location(target, self.safe_start[i])

        # get state
        # random camera
        self.cam_pose = []
        if self.reset_type >= 5:
            for i, cam in enumerate(self.cam_id):
                cam_loc = [np.random.randint(self.cam_area[i][0], self.cam_area[i][1]),
                           np.random.randint(self.cam_area[i][2], self.cam_area[i][3]),
                           np.random.randint(self.cam_area[i][4], self.cam_area[i][5])]
                self.unrealcv.set_location(cam, cam_loc)
                cam_rot = self.unrealcv.get_rotation(cam, 'hard')
                angle_h = self.get_direction(cam_loc+cam_rot, self.safe_start[0])
                angle_v = self.get_verti_direction(cam_loc+cam_rot, self.safe_start[0])
                cam_rot[1] += angle_h
                cam_rot[2] -= angle_v
                self.unrealcv.set_rotation(cam, cam_rot)
                self.cam_pose.append(cam_loc + cam_rot)
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

        self.states, self.obj_pos = self.unrealcv.get_pose_img_batch(self.player_list, self.cam_id, 'lit', 'bmp')

        self.count_steps = 0
        if 'Ram' in self.target or 'Nav' in self.target:
            for i in range(len(self.random_agents)):
                self.random_agents[i].reset()

        return self.states

    def close(self):
        self.unreal.close()

    def render(self, mode='rgb_array', close=False):
        if close==True:
            self.unreal.close()
        for i in range(self.num_cam):
            cv2.imshow('tracker_{}'.format(str(i)), self.states[i])
        cv2.waitKey(1)
        return self.unrealcv.img_color

    def seed(self, seed=None):
        self.num_cam = seed

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

    def get_start_area(self, safe_start, safe_range):
        start_area = [safe_start[0] - safe_range, safe_start[0] + safe_range,
                      safe_start[1] - safe_range, safe_start[1] + safe_range]
        return start_area
