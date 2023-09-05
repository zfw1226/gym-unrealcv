import math
import gym
import numpy as np
from gym import spaces
from gym_unrealcv.envs.tracking.baseline import *
from gym_unrealcv.envs.utils import env_unreal
from gym_unrealcv.envs.tracking.interaction import Tracking
import gym_unrealcv
import cv2
from gym_unrealcv.envs.utils.misc import *
from gym_unrealcv.envs.utils.visualization import *
import random
''' 
It is an env for multi-camera active object tracking.
State : raw color image
Action:  rotate cameras (yaw, pitch)
Task: Learn to follow the target object(moving person) with multiple cameras in the scene
'''


class UnrealCvMultiCam(gym.Env):
    def __init__(self,
                 setting_file,
                 reset_type,
                 action_type='discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 reward_type='distance',  # distance
                 docker=False,
                 resolution=(320, 240),
                 nav='Random',  # Random, Goal, Internal
                 ):

        self.docker = docker
        self.reset_type = reset_type
        self.roll = 0
        self.nav = nav
        setting = load_env_setting(setting_file)
        self.env_name = setting['env_name']
        self.cam_id = setting['cam_id']
        self.target_list = setting['targets']
        self.discrete_actions = setting['discrete_actions']
        self.discrete_actions_player = setting['discrete_actions_player']
        self.continous_actions = setting['continous_actions']
        self.continous_actions_player = setting['continous_actions_player']
        self.max_steps = setting['max_steps']
        self.max_obstacles = setting['max_obstacles']
        self.height = setting['height']
        self.pitch = setting['pitch']
        self.objects_env = setting['objects_list']
        if setting.get('reset_area'):
            self.reset_area = setting['reset_area']
        if setting.get('cam_area'):
            self.cam_area = setting['cam_area']

        self.test = False if 'MCRoom' in self.env_name else True

        if setting.get('goal_list'):
            self.goal_list = setting['goal_list']
        if setting.get('camera_loc'):
            self.camera_loc = setting['camera_loc']

        # parameters for rendering map
        self.target_move = setting['target_move']
        self.camera_move = setting['camera_move']
        self.scale_rate = setting['scale_rate']
        self.pose_rate = setting['pose_rate']

        self.background_list = setting['backgrounds']
        self.light_list = setting['lights']
        self.target_num = setting['target_num']
        texture_dir = setting['imgs_dir']
        gym_path = os.path.dirname(gym_unrealcv.__file__)
        texture_dir = os.path.join(gym_path, 'envs', 'UnrealEnv', texture_dir)
        self.textures_list = os.listdir(texture_dir)
        self.safe_start = setting['safe_start']
        self.start_area = self.get_start_area(self.safe_start[0], 100)

        self.num_target = setting['target_num']
        self.resolution = resolution

        for i in range(len(self.textures_list)):
            if self.docker:
                self.textures_list[i] = os.path.join('/unreal', setting['imgs_dir'], self.textures_list[i])
            else:
                self.textures_list[i] = os.path.join(texture_dir, self.textures_list[i])

        # start unreal env
        self.unreal = env_unreal.RunUnreal(ENV_BIN=setting['env_bin'], ENV_MAP=setting['env_map'])
        env_ip, env_port = self.unreal.start(docker, resolution)

        # connect UnrealCV
        self.unrealcv = Tracking(cam_id=0, port=env_port, ip=env_ip,
                                 env=self.unreal.path2env, resolution=resolution)


        # remove or add the target
        while len(self.target_list) > self.target_num:
            self.unrealcv.destroy_obj(self.target_list.pop())
            # self.cam_id.pop()
        while len(self.target_list) < self.target_num:
            name = 'target_C_{0}'.format(len(self.player_list)+1)
            if name in self.freeze_list:
                self.freeze_list.remove(name)
            else:
                self.unrealcv.new_obj('target_C', name, random.sample(self.safe_start, 1)[0])
            self.unrealcv.set_obj_color(name, np.random.randint(0, 255, 3))
            self.unrealcv.set_random(name, 0)
            self.target_list.append(name)
            # self.unrealcv.set_interval(self.interval, name)

        # remove or add the camera
        while len(self.cam_id) < setting['max_cam_num']:
            self.unrealcv.new_camera()
            self.cam_id.append(self.unrealcv.get_camera_num()-1)

        while len(self.cam_id) > setting['max_cam_num']:
            self.unrealcv.destroy_obj(self.cam_id.pop())

        self.num_cam = len(self.cam_id)
        self.cam_height = [setting['height'] for i in range(self.num_cam)]
        print(self.cam_id, self.target_list)
        self.unrealcv.color_dict = self.unrealcv.build_color_dic(self.target_list)
        for obj in self.unrealcv.get_objects():
            if obj not in self.target_list:
                self.unrealcv.set_obj_color(obj, [0, 0, 0])

        # define action
        self.action_type = action_type
        assert self.action_type == 'Discrete' or self.action_type == 'Continuous'
        if self.action_type == 'Discrete':
            self.action_space = [spaces.Discrete(len(self.discrete_actions)) for i in range(self.num_cam)]
            player_action_space = spaces.Discrete(len(self.discrete_actions_player))
        elif self.action_type == 'Continuous':
            self.action_space = [spaces.Box(low=np.array(self.continous_actions['low']),
                                            high=np.array(self.continous_actions['high'])) for i in range(self.num_cam)]
            player_action_space = spaces.Box(low=np.array(self.continous_actions_player['low']),
                                             high=np.array(self.continous_actions_player['high']))

        self.count_steps = 0

        # define observation space,
        # color, depth, rgbd, ...
        self.observation_type = observation_type
        assert self.observation_type in ['Color', 'Depth', 'Rgbd', 'Gray']
        self.observation_space = [self.unrealcv.define_observation(self.cam_id[i], self.observation_type, 'fast')
                                  for i in range(self.num_cam)]

        self.unrealcv.pitch = self.pitch
        # define reward type
        # distance, bbox, bbox_distance
        self.reward_type = reward_type
        # self.reward_function = reward.Reward(setting)

        if not self.test:
            if self.reset_type >= 0:
                self.unrealcv.init_objects(self.objects_env)

        self.count_close = 0

        self.person_id = 0
        self.unrealcv.set_location(0, [self.safe_start[0][0], self.safe_start[0][1], self.safe_start[0][2]+600])
        self.unrealcv.set_rotation(0, [0, -180, -90])

        if 'Random' in self.nav:
            self.random_agents = [RandomAgent(player_action_space) for i in range(self.num_target)]
        if 'Goal' in self.nav:
            if not self.test:
                self.random_agents = [GoalNavAgent(self.continous_actions_player, self.reset_area, 'Mid') for i in range(self.num_target)]
            else:
                self.random_agents = [GoalNavAgentTest(self.continous_actions_player, goal_list=self.goal_list)
                                      for i in range(self.num_target)]

        self.unrealcv.set_interval(30)

        self.record_eps = 0
        self.min_mask_area = 0.01  # the min area of the target in the image

    def step(self, actions):
        info = dict(
            Done=False,
            Reward=[0 for i in range(self.num_cam)],
            Target_Pose=[],
            Cam_Pose=[],
            Steps=self.count_steps,
        )

        self.current_cam_pos = self.cam_pose.copy()
        self.current_target_pos = self.target_pos.copy()

        # take actions on cameras
        actions = np.squeeze(actions)
        actions2cam = []
        for i in range(self.num_cam):
            if self.action_type == 'Discrete':
                actions2cam.append(self.discrete_actions[actions[i]])  # delta_yaw, delta_pitch
            else:
                actions2cam.append(actions[i])  # delta_yaw, delta_pitch

        # move the target
        actions2target = []
        for i in range(len(self.target_list)):
            if 'Random' in self.nav:
                if self.action_type == 'Discrete':
                    actions2target.append(self.discrete_actions_player[self.random_agents[i].act(self.target_pos[i])])
                else:
                    actions2target.append(self.random_agents[i].act(self.target_pos[i]))
            if 'Goal' in self.nav:
                    actions2target.append(self.random_agents[i].act(self.target_pos[i]))
        for i, target in enumerate(self.target_list):
            self.unrealcv.set_move(target, actions2target[i][1], actions2target[i][0])

        self.gate_ids = np.zeros(len(self.cam_id), int)
        self.gt_actions = []

        self.gate_gt_ids = np.ones(len(self.cam_id), int)

        for i, cam in enumerate(self.cam_id):
            cam_rot = self.unrealcv.get_rotation(cam, 'hard')
            # take actions on cameras
            cam_rot[1] += actions2cam[i][0] * self.zoom_scales[i]
            cam_rot[2] += actions2cam[i][1] * self.zoom_scales[i]
            cam_rot[2] = np.clip(cam_rot[2], -85, 85)
            self.unrealcv.set_rotation(cam, cam_rot)
            self.cam_pose[i][-3:] = cam_rot
            self.unrealcv.adjust_fov(cam, actions2cam[i][-1])

        # use batch command to get data from unrealcv
        self.states = self.unrealcv.get_img_batch(self.cam_id, 'lit')

        # get object_masks will cost a lot of time in large scene
        # TODO use distance-based reward instead
        object_masks = self.unrealcv.get_img_batch(self.cam_id, 'object_mask')
        for i, obj_mask in enumerate(object_masks):
            for bbox in self.unrealcv.get_bboxes(obj_mask, self.target_list):
                self.gate_ids[i] += self.check_visibility(bbox, self.min_mask_area)

        # imgs = np.hstack([mask for mask in object_masks])
        # cv2.imshow('object_mask', imgs)
        # cv2.waitKey(1)

        self.count_steps += 1

        rewards = []
        gt_locations = []

        expected_scales = []
        hori_rewards = []
        verti_rewards = []

        cal_target_observed = np.zeros(len(self.cam_id))
        self.target_observed = np.zeros(len(self.cam_id))

        # get bbox and reward
        for i, cam in enumerate(self.cam_id):
            # get relative location and reward
            max_hori_angle = self.unrealcv.cam[cam]['fov']/2
            max_verti_angle = self.unrealcv.cam[cam]['fov']*self.resolution[1]/(2*self.resolution[0])
            direction = get_direction(self.cam_pose[i], self.target_pos[0])
            hori_reward = 1 - 2*abs(direction) / max_hori_angle
            verti_direction = self.get_verti_direction(self.cam_pose[i], self.target_pos[0])
            verti_reward = 1 - 2*abs(verti_direction) / max_verti_angle

            hori_rewards.append(hori_reward)
            verti_rewards.append(verti_reward)
            pose_reward = max(hori_reward+verti_reward, -2) / 2

            # TODO: add scale reward
            d = self.unrealcv.get_distance(self.cam_pose[i], self.target_pos[0], 3)
            gt_locations.append([direction, verti_direction, d])
            expected_scale = self.scale_function(d)
            # zoom_error = abs(self.zoom_scales[i] - expected_scale) / (1 - self.min_scale)
            zoom_reward = 0
            expected_scales.append(expected_scale)

            if abs(direction) <= max_hori_angle and abs(verti_direction) <= max_verti_angle:
                cal_target_observed[i] = 1
                sparse_reward = pose_reward + zoom_reward if self.gate_ids[i] != 0 else 0
            else:
                sparse_reward = -1

            reward = sparse_reward
            rewards.append(reward)
        info['gt_locations'] = gt_locations
        info['gate rewards'] = self.gate_ids
        info['camera poses'] = self.current_cam_pos

        if self.count_steps > self.max_steps:
            info['Done'] = True

        if not self.test:
            if sum(cal_target_observed) < 2:
                self.count_close += 1
            else:
                self.count_close = 0

            if self.count_close > 10:
                info['Done'] = True

        if info['Done']:
            self.record_eps += 1

        info['states'] = self.states
        info['gate ids'] = self.gate_ids
        info['Reward'] = rewards
        info['Success rate'] = sum(cal_target_observed) / self.num_cam
        info['Success ids'] = cal_target_observed

        for i, target in enumerate(self.target_list):
            self.target_pos[i] = self.unrealcv.get_obj_pose(target)

        return self.states, info['Reward'], info['Done'], info

    def reset(self):
        self.zoom_scales = np.ones(self.num_cam)
        self.C_reward = 0
        self.count_close = 0

        self.target_pos = np.array([np.random.randint(self.start_area[0], self.start_area[1]),
                                    np.random.randint(self.start_area[2], self.start_area[3]),
                                    self.safe_start[0][-1]])

        for i, target in enumerate(self.target_list):
            self.unrealcv.set_move(target, self.safe_start[i][0], self.safe_start[i][1])

        # environment augmentation
        if self.reset_type >= 1:
            if self.env_name == 'MCRoom':  # training env
                map_id = [2, 3, 6, 7, 9]
                spline = False
                object_app = np.random.choice(map_id)
            else:  # testing envs
                map_id = [6, 7, 9, 3]
                spline = True
                object_app = map_id[self.person_id % len(map_id)]
                self.person_id += 1

            self.unrealcv.set_appearance(self.target_list[0], object_app, spline)
            self.obstacles_num = self.max_obstacles
            self.obstacle_scales = [[1, 1.2] if np.random.binomial(1, 0.5) == 0 else [1.5, 2] for k in range(self.obstacles_num)]
            self.unrealcv.clean_obstacles()
            if not self.test:
                self.unrealcv.random_obstacles(self.objects_env, self.textures_list,
                                           self.obstacles_num, self.reset_area, self.start_area, self.obstacle_scales)
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
                    self.unrealcv.set_light(lit, lit_direction, np.random.uniform(1, 4), np.random.uniform(0.1, 1, 3))

        # target appearance
        if self.reset_type >= 3:
            self.unrealcv.random_player_texture(self.target_list[0], self.textures_list, 3)
            self.unrealcv.random_texture(self.background_list, self.textures_list, 5)

        # texture
        if self.reset_type >= 4 and not self.test:
            self.obstacle_scales = [[1, 1.5] if np.random.binomial(1, 0.5) == 0 else [2, 2.5] for k in  # 2.5-3.5 before
                                    range(self.obstacles_num)]
            self.obstacles_num = self.max_obstacles
            self.unrealcv.clean_obstacles()
            if not self.test:
                self.unrealcv.random_obstacles(self.objects_env, self.textures_list,
                                           self.obstacles_num, self.reset_area, self.start_area, self.obstacle_scales)

        self.target_pos = []
        for i, target in enumerate(self.target_list):
            self.unrealcv.set_obj_location(target, self.safe_start[i])
            self.target_pos.append(self.unrealcv.get_obj_pose(target))


        # reset cameras
        states = []
        self.cam_pose = []
        self.fixed_cam = True if self.test else False
        self.gt_actions = []
        self.gate_ids = np.zeros(len(self.cam_id), int)

        for i, cam in enumerate(self.cam_id):
            if self.test:
                cam_loc = self.camera_loc[i]
                self.unrealcv.set_location(cam, cam_loc)
                cam_rot = self.unrealcv.get_rotation(cam, 'hard')
            else:
                cam_loc = [np.random.randint(self.cam_area[i][0], self.cam_area[i][1]),
                           np.random.randint(self.cam_area[i][2], self.cam_area[i][3]),
                           np.random.randint(self.cam_area[i][4], self.cam_area[i][5])]
                cam_rot = self.unrealcv.get_rotation(cam, 'hard')

            angle_h = get_direction(cam_loc + cam_rot, self.target_pos[0])
            angle_v = self.get_verti_direction(cam_loc + cam_rot, self.target_pos[0])
            cam_rot[1] += angle_h
            cam_rot[2] -= angle_v

            self.unrealcv.set_location(cam, cam_loc)
            self.unrealcv.set_rotation(cam, cam_rot)
            if self.unrealcv.get_fov(cam) != 90:  # set the camera fov to 90
                self.unrealcv.set_fov(cam, 90)
            self.cam_pose.append(cam_loc+cam_rot)

            raw_state = self.unrealcv.get_observation(cam, self.observation_type, 'fast')
            states.append(raw_state)

            object_mask = self.unrealcv.read_image(self.cam_id[i], 'object_mask', 'fast')
            for bbox in self.unrealcv.get_bboxes(object_mask, self.target_list):
                self.gate_ids[i] += self.check_visibility(bbox, self.min_mask_area)  # check if the target is visible

        self.count_steps = 0
        if 'Random' in self.nav or 'Goal' in self.nav:
            for i in range(len(self.random_agents)):
                self.random_agents[i].reset()

        self.states = states
        self.current_cam_pos = self.cam_pose.copy()
        self.current_target_pos = self.target_pos.copy()

        return np.array(self.states)

    def close(self):
        self.unrealcv.client.disconnect()
        self.unreal.close()

    def render(self, mode='rgb_array', close=False):
        if close==True:
            self.unreal.close()
        imgs = np.hstack([s for s in self.states])
        # imgs = np.hstack([self.states[0], self.states[1], self.states[2], self.states[3]])
        return imgs

    def to_render(self, choose_ids):
        map_render(self.cam_pose, self.target_pos[0], choose_ids, self.target_move, self.camera_move, self.scale_rate,
                   self.pose_rate)

    def seed(self, seed=None):
        self.person_id = seed

    def get_action_size(self):
        return len(self.action)

    def get_start_area(self, safe_start, safe_range):
        start_area = [safe_start[0] - safe_range, safe_start[0] + safe_range,
                      safe_start[1] - safe_range, safe_start[1] + safe_range]
        return start_area

    def scale_function(self, d):
        scale = 200 / d  # small camera area
        return scale

    def get_verti_direction(self, current_pose, target_pose):
        person_height = target_pose[2]
        plane_distance = self.unrealcv.get_distance(current_pose, target_pose, 2)
        height = current_pose[2] - person_height
        angle = np.arctan2(height, plane_distance) / np.pi * 180
        angle_now = angle + current_pose[-1]
        return angle_now

    def check_visibility(self, bbox, min_size):
        w = bbox[1][0] - bbox[0][0]
        h = bbox[1][1] - bbox[0][1]
        area = w * h
        if area < min_size:
            return 0
        else:
            return 1

