import math
import gym
from gym import spaces
from gym_unrealcv.envs.tracking.baseline import *
from gym_unrealcv.envs.utils import env_unreal
from gym_unrealcv.envs.tracking.interaction import Tracking
import gym_unrealcv
import cv2
import matplotlib.pyplot as plt
from gym_unrealcv.envs.utils.misc import *

''' 
It is an env for multi-camera active object tracking.
State : raw color image
Action:  rotate cameras (yaw, pitch)
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
            player_action_space = [spaces.Discrete(len(self.discrete_actions_player)) for i in range(1)]
        elif self.action_type == 'Continuous':
            self.action_space = [spaces.Box(low=np.array(self.continous_actions['low']),
                                            high=np.array(self.continous_actions['high'])) for i in range(self.num_cam)]
            player_action_space = spaces.Discrete(len(self.continous_actions_player))

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
        self.unrealcv.set_obj_location("TargetBP", [-3000, -3000, 220])  # remove the additional target
        if 'Random' in self.nav:
            self.random_agents = [RandomAgent(self.continous_actions_player) for i in range(self.num_target)]
        if 'Goal' in self.nav:
            if not self.test:
                self.random_agents = [GoalNavAgent(self.continous_actions_player, self.reset_area, 'Mid') for i in range(self.num_target)]
            else:
                self.random_agents = [GoalNavAgentTest(self.continous_actions_player, goal_list=self.goal_list)
                                      for i in range(self.num_target)]

        self.unrealcv.set_interval(30)

        self.record_eps = 0
        self.max_mask_area = np.ones(self.num_cam) * 0.001 * self.resolution[0] * self.resolution[1]

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
        self.current_states = self.states

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
                    actions2target.append(self.random_agents[i].act())
                else:
                    actions2target.append(self.random_agents[i].act())
            if 'Goal' in self.nav:
                    actions2target.append(self.random_agents[i].act(self.target_pos[i]))
        for i, target in enumerate(self.target_list):
            self.unrealcv.set_move(target, actions2target[i][1], actions2target[i][0])

        states = []
        self.gate_ids = []
        self.gt_actions = []

        self.gate_gt_ids = np.ones(len(self.cam_id), int)

        for i, cam in enumerate(self.cam_id):
            cam_rot = self.unrealcv.get_rotation(cam, 'hard')
            if len(actions2cam[i]) == 2:
                cam_rot[1] += actions2cam[i][0] * self.zoom_scales[i]
                cam_rot[2] += actions2cam[i][1] * self.zoom_scales[i]

                cam_rot[2] = cam_rot[2] if cam_rot[2] < 80.0 else 80.0
                cam_rot[2] = cam_rot[2] if cam_rot[2] > - 80.0 else -80.0

                self.unrealcv.set_rotation(cam, cam_rot)
                self.cam_pose[i][-3:] = cam_rot
                raw_state = self.unrealcv.get_observation(cam, self.observation_type, 'fast')

                zoom_state = raw_state[int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2): (
                            self.resolution[1] - int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2)),
                             int(self.resolution[0] * (1 - self.zoom_scales[i]) / 2): (self.resolution[0] - int(self.resolution[0] * (1 - self.zoom_scales[i]) / 2)),:]
                state = cv2.resize(zoom_state, self.resolution)

            else:  # zoom action
                raw_state = self.unrealcv.get_observation(cam, self.observation_type, 'fast')
                if actions2cam[i][0] == 1:  # zoom in
                    self.zoom_scales[i] = self.zoom_in_scale * self.zoom_scales[i] if self.zoom_in_scale * self.zoom_scales[i] >= self.min_scale else self.zoom_scales[i]
                    zoom_state = raw_state[int(self.resolution[1]*(1-self.zoom_scales[i])/2): (self.resolution[1] -
                            int(self.resolution[1]*(1-self.zoom_scales[i])/2)), int(self.resolution[0]*(1-self.zoom_scales[i])/2): (self.resolution[0] -
                                                                                    int(self.resolution[0]*(1-self.zoom_scales[i])/2)), :]
                    state = cv2.resize(zoom_state, self.resolution)
                elif actions2cam[i][0] == -1:  # zoom out
                    self.zoom_scales[i] = self.zoom_out_scale * self.zoom_scales[i] if self.zoom_out_scale * self.zoom_scales[i] <= 1 else self.zoom_scales[i]
                    zoom_state = raw_state[int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2): (self.resolution[1] - int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2)),
                                 int(self.resolution[0] * (1 - self.zoom_scales[i]) / 2): (self.resolution[0] -int(self.resolution[0] * (1 - self.zoom_scales[i]) / 2)),:]
                    state = cv2.resize(zoom_state, self.resolution)
                else:
                    zoom_state = raw_state[int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2): ( self.resolution[1] - int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2)),
                                 int(self.resolution[0] * (1 - self.zoom_scales[i]) / 2): (self.resolution[0] - int(self.resolution[0] * (1 - self.zoom_scales[i]) / 2)),:]
                    state = cv2.resize(zoom_state, self.resolution)


            # get visibility gt for training gate
            object_mask = self.unrealcv.read_image(self.cam_id[i], 'object_mask', 'fast')
            zoom_mask = object_mask[int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2): (
                    self.resolution[1] - int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2)),
                    int(self.resolution[0] * (1 - self.zoom_scales[i]) / 2): (self.resolution[0] - int(
                        self.resolution[0] * (1 - self.zoom_scales[i]) / 2)), :]
            zoom_mask = cv2.resize(zoom_mask, self.resolution)
            bbox = self.unrealcv.get_bboxes(zoom_mask, self.target_list)
            w = self.resolution[0] * (bbox[0][1][0] - bbox[0][0][0])
            h = self.resolution[1] * (bbox[0][1][1] - bbox[0][0][1])
            area = w * h
            if area <= self.max_mask_area[i]:
                self.gate_ids.append(0)
            else:
                self.gate_ids.append(1)
            states.append(state)
            self.unrealcv.set_rotation(cam, cam_rot)

        self.states = states
        self.count_steps += 1

        obj_masks = []

        rewards = []
        gt_locations = []

        expected_scales = []
        hori_rewards = []
        verti_rewards = []

        cal_target_observed = np.zeros(len(self.cam_id))
        self.target_observed = np.zeros(len(self.cam_id))

        # get bbox and reward
        for i in range(len(self.cam_id)):
            # get bbox
            if self.reset_type >= 6:
                object_mask = self.unrealcv.read_image(self.cam_id[i], 'object_mask', 'fast')
                bbox = self.unrealcv.get_bboxes(object_mask, self.target_list)
                # get bbox size
                bbox_shape = np.array(bbox[0][1]) - np.array(bbox[0][0])
                if bbox_shape[0] * bbox_shape[1] < 0.01:
                    self.target_observed[i] = 1
                obj_masks.append(object_mask)

            # get relative location and reward
            direction = get_direction(self.cam_pose[i], self.target_pos[0])
            hori_reward = 1 - 2*abs(direction) / 45.0

            verti_direction = self.get_verti_direction(self.cam_pose[i], self.target_pos[0])
            verti_reward = 1 - 2*abs(verti_direction) / 30.0

            hori_rewards.append(hori_reward)
            verti_rewards.append(verti_reward)

            d = self.unrealcv.get_distance(self.cam_pose[i], self.target_pos[0], 3)

            gt_locations.append([direction, verti_direction, d])

            expected_scale = self.scale_function(d)
            expected_scale = expected_scale if expected_scale >= self.min_scale else self.min_scale
            expected_scale = expected_scale if expected_scale <= 1 else 1
            zoom_error = abs(self.zoom_scales[i] - expected_scale) / (1 - self.min_scale)
            zoom_reward = 1 - zoom_error
            expected_scales.append(expected_scale)
            pose_reward = max(2 - abs(direction) / 45.0 - abs(verti_direction) / 30.0, -2) / 2

            if abs(direction) <= 45.0 * self.zoom_scales[i] and abs(verti_direction) <= 30.0 * self.zoom_scales[i]:
                cal_target_observed[i] = 1
                sparse_reward = pose_reward + zoom_reward if self.gate_ids[i] != 0 else 0
            else:
                sparse_reward = -1

            reward = sparse_reward
            rewards.append(reward)
        info['gt_locations'] = gt_locations
        info['gate rewards'] = self.gate_rewards
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

        return np.array(self.states), info['Reward'], info['Done'], info

    def reset(self):

        self.zoom_scales = np.ones(self.num_cam)
        self.zoom_in_scale = 0.9
        self.zoom_out_scale = 1.1
        self.stand_d = 500
        self.min_scale = 0.3
        self.limit_scales = np.zeros(self.num_cam)

        self.C_reward = 0
        self.count_close = 0

        self.target_pos = np.array([np.random.randint(self.start_area[0], self.start_area[1]),
                                    np.random.randint(self.start_area[2], self.start_area[3]),
                                    self.safe_start[0][-1]])

        for i, target in enumerate(self.target_list):
            self.unrealcv.set_move(target, self.safe_start[i][0], self.safe_start[i][1])

        if self.reset_type >= 1:
            if self.env_name == 'MCRoom':
                map_id = [2, 3, 6, 7, 9]
                spline = False
                object_app = np.random.choice(map_id)
            else:
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
                    self.unrealcv.set_light(lit, lit_direction, np.random.uniform(1, 4), np.random.uniform(0.1,1,3))

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

        states = []
        self.cam_pose = []
        self.fixed_cam = True if self.test else False
        self.gt_actions = []
        self.gate_rewards = []
        self.gate_ids = []

        if self.reset_type >= 5:
            for i, cam in enumerate(self.cam_id):
                if self.fixed_cam:
                    cam_loc = self.camera_loc[i]
                else:
                    cam_loc = [np.random.randint(self.cam_area[i][0], self.cam_area[i][1]),
                               np.random.randint(self.cam_area[i][2], self.cam_area[i][3]),
                               np.random.randint(self.cam_area[i][4], self.cam_area[i][5])]

                self.unrealcv.set_location(cam, cam_loc)
                cam_rot = self.unrealcv.get_rotation(cam, 'hard')
                angle_h = get_direction(cam_loc+cam_rot, self.target_pos[0])
                angle_v = self.get_verti_direction(cam_loc+cam_rot, self.target_pos[0])
                cam_rot[1] += angle_h
                cam_rot[2] -= angle_v

                self.unrealcv.set_rotation(cam, cam_rot)
                self.cam_pose.append(cam_loc + cam_rot)

                raw_state = self.unrealcv.get_observation(cam, self.observation_type, 'fast')

                object_mask = self.unrealcv.read_image(self.cam_id[i], 'object_mask', 'fast')
                zoom_mask = object_mask[int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2): (
                        self.resolution[1] - int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2)),
                            int(self.resolution[0] * (1 - self.zoom_scales[i]) / 2): (self.resolution[0] - int(
                                self.resolution[0] * (1 - self.zoom_scales[i]) / 2)), :]
                zoom_mask = cv2.resize(zoom_mask, self.resolution)
                bbox = self.unrealcv.get_bboxes(zoom_mask, self.target_list)
                w = self.resolution[0] * (bbox[0][1][0] - bbox[0][0][0])
                h = self.resolution[1] * (bbox[0][1][1] - bbox[0][0][1])
                area = w * h

                if area <= self.max_mask_area[i] * self.zoom_scales[i]:
                    self.gate_ids.append(0)
                else:
                    self.gate_ids.append(1)

                states.append(raw_state)
                sparse_reward = 0 if area <= self.max_mask_area[i] else 1
                self.gate_rewards.append(sparse_reward)

        else:
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
                self.cam_pose.append(cam_loc+cam_rot)

                raw_state = self.unrealcv.get_observation(cam, self.observation_type, 'fast')
                states.append(raw_state)

                object_mask = self.unrealcv.read_image(self.cam_id[i], 'object_mask', 'fast')
                bbox = self.unrealcv.get_bboxes(object_mask, self.target_list)
                w = self.resolution[0] * (bbox[0][1][0] - bbox[0][0][0])
                h = self.resolution[1] * (bbox[0][1][1] - bbox[0][0][1])
                area = w * h

                if area <= self.max_mask_area[i] * self.zoom_scales[i]:
                    self.gate_ids.append(0)
                else:
                    self.gate_ids.append(1)

                sparse_reward = 0 if area <= self.max_mask_area[i] else 1
                self.gate_rewards.append(sparse_reward)

        self.count_steps = 0
        if 'Random' in self.nav or 'Goal' in self.nav:
            for i in range(len(self.random_agents)):
                self.random_agents[i].reset()

        self.current_states = self.states = states
        self.current_cam_pos = self.cam_pose.copy()
        self.current_target_pos = self.target_pos.copy()

        return np.array(self.states)

    def close(self):
        self.unreal.close()

    def render(self, mode='rgb_array', close=False):
        if close==True:
            self.unreal.close()
        imgs = np.hstack([self.states[0], self.states[1], self.states[2], self.states[3]])
        cv2.imshow("Pose-assisted-multi-camera-collaboration", imgs)
        cv2.waitKey(1)

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


def map_render(camera_pos, target_pos, choose_ids, target_move, camera_move, scale_rate, pose_rate):

    length = 600
    coordinate_delta = np.mean(np.array(camera_pos)[:, :2], axis=0)
    img = np.zeros((length + 1, length + 1, 3)) + 255
    num_cam = len(camera_pos)

    camera_position_origin = np.array([camera_pos[i][:2] for i in range(num_cam)])
    target_position_origin = np.array(target_pos[:2])

    lengths = []
    for i in range(num_cam):
        length = np.sqrt(sum(np.array(camera_position_origin[i] - coordinate_delta)) ** 2)
        lengths.append(length)
    pose_scale = max(lengths)

    pose_scale = pose_scale * pose_rate
    target_position = length * (np.array([scale_rate + (target_position_origin[0] - coordinate_delta[0]) / pose_scale, scale_rate +
                                          (target_position_origin[1] - coordinate_delta[0]) / pose_scale])) / 2 + np.array(target_move)

    camera_position = []
    for i in range(num_cam):
        position_transfer = length * (np.array([scale_rate + (camera_position_origin[i][0] - coordinate_delta[0]) / pose_scale,
                                                scale_rate + (camera_position_origin[i][1] - coordinate_delta[1]) / pose_scale])) / 2 + np.array(camera_move)
        camera_position.append(position_transfer)

    abs_angles = [camera_pos[i][4] for i in range(num_cam)]

    color_dict = {'red': [255, 0, 0], 'black': [0, 0, 0], 'blue': [0, 0, 255], 'green': [0, 255, 0],
                  'darkred': [128, 0, 0], 'yellow': [255, 255, 0], 'deeppink': [255, 20, 147]}

    # plot camera
    for i in range(num_cam):
        img[int(camera_position[i][1])][int(camera_position[i][0])][0] = color_dict["black"][0]
        img[int(camera_position[i][1])][int(camera_position[i][0])][1] = color_dict["black"][1]
        img[int(camera_position[i][1])][int(camera_position[i][0])][2] = color_dict["black"][2]

    # plot target
    img[int(target_position[1])][int(target_position[0])][0] = color_dict['blue'][0]
    img[int(target_position[1])][int(target_position[0])][1] = color_dict['blue'][1]
    img[int(target_position[1])][int(target_position[0])][2] = color_dict['blue'][2]

    plt.cla()
    plt.imshow(img.astype(np.uint8))

    # get camera's view space positions
    visua_len = 60
    for i in range(num_cam):
        theta = abs_angles[i] + 90.0
        dx = visua_len * math.sin(theta * math.pi / 180)
        dy = - visua_len * math.cos(theta * math.pi / 180)
        plt.arrow(camera_position[i][0], camera_position[i][1], dx, dy, width=0.1, head_width=8,
                  head_length=8, length_includes_head=True)

        plt.annotate(str(i), xy=(camera_position[i][0], camera_position[i][1]),
                     xytext=(camera_position[i][0], camera_position[i][1]), fontsize=10, color='blue')

        # top-left
        if int(choose_ids[i]) == 0:
            plt.annotate('cam {0} use pose'.format(i), xy=(camera_position[i][0], camera_position[i][1]),  xytext=(350, (1 + i) * 50 + 250), fontsize=10, color='red')
        else:
            plt.annotate('cam {0} use vision'.format(i), xy=(camera_position[i][0], camera_position[i][1]), xytext=(350, (1 + i) * 50 + 250), fontsize=10,
                         color='blue')

    plt.plot(target_position[0], target_position[1], 'ro')
    plt.title("Top-view")
    plt.xticks([])
    plt.yticks([])
    plt.pause(0.01)
