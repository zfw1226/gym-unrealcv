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
import random
import sys
''' 
It is an env for active object tracking.

State : raw color image and depth
Action:  (linear velocity ,angle velocity) 
Done : the relative distance or angle to target is larger than the threshold.
Task: Learn to follow the target object(moving person) in the scene
'''

# 0: tracker 1:target 2~n:others
# cam_id  0:global 1:tracker 2:target 3:others
class UnrealCvTracking_1vn(gym.Env):
    def __init__(self,
                 setting_file,
                 reset_type=0,
                 action_type='Discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 reward_type='distance',  # distance
                 docker=False,
                 resolution=(320, 240),
                 target='Nav',  # Ram, Nav, Internal
                 ):
        self.docker = docker
        self.reset_type = reset_type
        self.roll = 0
        self.target = target
        setting = misc.load_env_setting(setting_file)
        self.env_name = setting['env_name']
        self.cam_id = setting['cam_id']
        self.player_list = setting['players']
        self.discrete_actions = setting['discrete_actions']
        self.discrete_actions_player = setting['discrete_actions_player']
        self.continous_actions = setting['continous_actions']
        self.continous_actions_player = setting['continous_actions_player']
        self.max_steps = setting['max_steps']
        self.max_distance = setting['max_distance']
        self.min_distance = setting['min_distance']
        self.max_direction = setting['max_direction']
        self.height = setting['height']
        self.pitch = setting['pitch']
        self.objects_list = setting['objects_list']
        self.reset_area = setting['reset_area']
        self.background_list = setting['backgrounds']
        self.light_list = setting['lights']
        self.max_player_num = setting['max_player_num']  # the max players number
        self.exp_distance = setting['exp_distance']
        texture_dir = setting['imgs_dir']
        gym_path = os.path.dirname(gym_unrealcv.__file__)
        texture_dir = os.path.join(gym_path, 'envs', 'UnrealEnv', texture_dir)
        self.textures_list = os.listdir(texture_dir)
        self.safe_start = setting['safe_start']
        self.interval = setting['interval']
        self.start_area = self.get_start_area(self.safe_start[0], 500)
        self.top = False
        self.person_id = 0
        self.count_eps = 0
        self.count_steps = 0
        self.count_close = 0
        self.direction = None
        self.freeze_list = []
        self.resolution = resolution

        for i in range(len(self.textures_list)):
            if self.docker:
                self.textures_list[i] = os.path.join('/unreal', setting['imgs_dir'], self.textures_list[i])
            else:
                self.textures_list[i] = os.path.join(texture_dir, self.textures_list[i])

        # start unreal env
        if 'linux' in sys.platform:
            env_bin = setting['env_bin']
        elif 'win' in sys.platform:
            env_bin = setting['env_bin_win']
        self.unreal = env_unreal.RunUnreal(ENV_BIN=env_bin)
        env_ip, env_port = self.unreal.start(docker, resolution)

        # connect UnrealCV
        self.unrealcv = Tracking(cam_id=self.cam_id[0], port=env_port, ip=env_ip,
                                 env=self.unreal.path2env, resolution=resolution)

        # define action
        self.action_type = action_type
        assert self.action_type == 'Discrete' or self.action_type == 'Continuous'
        if self.action_type == 'Discrete':
            self.action_space = [spaces.Discrete(len(self.discrete_actions)) for i in range(self.max_player_num)]
            player_action_space = spaces.Discrete(len(self.discrete_actions_player))
            self.discrete_actions = np.array(self.discrete_actions)
            self.discrete_actions_player = np.array(self.discrete_actions_player)
        elif self.action_type == 'Continuous':
            self.action_space = [spaces.Box(low=np.array(self.continous_actions['low']),
                                      high=np.array(self.continous_actions['high'])) for i in range(self.max_player_num)]
            player_action_space = spaces.Discrete(len(self.continous_actions_player))

        self.count_steps = 0

        # define observation space,
        # color, depth, rgbd,...
        self.observation_type = observation_type
        assert self.observation_type in ['Color', 'Depth', 'Rgbd', 'Gray', 'CG', 'Mask']
        self.observation_space = [self.unrealcv.define_observation(self.cam_id[0], self.observation_type, 'fast')
                                  for i in range(self.max_player_num)]
        self.unrealcv.pitch = self.pitch

        # define reward type: distance
        self.reward_type = reward_type
        self.rendering = False

        if self.reset_type >= 4:
            self.unrealcv.init_objects(self.objects_list)

        self.count_close = 0
        self.unrealcv.set_random(self.player_list[0], 0)
        self.unrealcv.set_random(self.player_list[1], 0)

        self.person_id = 0
        if 'Ram' in self.target:
            self.random_agents = [baseline.RandomAgent(player_action_space) for i in range(self.max_player_num)]
        elif 'Nav' in self.target:
            self.random_agents = [baseline.GoalNavAgent(self.continous_actions_player, self.reset_area, self.target, 0
                                                        ) for i in range(self.max_player_num)]

        for player in self.player_list:
            self.unrealcv.set_interval(self.interval, player)
        self.unrealcv.build_color_dic(self.player_list)
        self.player_num = self.max_player_num
        self.action_factor = np.array([1.0, 1.0])
        self.smooth_factor = 0.6
        self.random_height = False
        self.early_stop = True
        self.get_bbox = False
        self.bbox = []

    def step(self, actions):
        info = dict(
            Collision=0,
            Done=False,
            Trigger=0.0,
            Reward=0.0,
            Action=actions,
            Pose=[],
            Steps=self.count_steps,
            Direction=None,
            Distance=None,
            Color=None,
            Depth=None,
            Relative_Pose=[]
        )
        actions2player = []
        for i in range(len(self.player_list)):
            if i < self.controable_agent:
                if self.action_type == 'Discrete':
                    act_now = self.discrete_actions[actions[i]]*self.action_factor
                    self.act_smooth[i] = self.act_smooth[i]*self.smooth_factor + act_now*(1-self.smooth_factor)
                    actions2player.append(self.act_smooth[i])
                else:
                    actions2player.append(actions[i])
            else:
                if 'Ram' in self.target:
                    if self.action_type == 'Discrete':
                        actions2player.append(self.discrete_actions_player[self.random_agents[i].act(self.obj_pos[i])])
                    else:
                        actions2player.append(self.random_agents[i].act(self.obj_pos[i]))
                if 'Nav' in self.target:
                    if i == 1:
                        actions2player.append(self.random_agents[i].act(self.obj_pos[i])*self.action_factor)
                    else:
                        actions2player.append(self.random_agents[i].act(self.obj_pos[i], self.random_agents[1].goal)*self.action_factor)

        self.unrealcv.set_move_batch(self.player_list, actions2player)
        self.count_steps += 1

        # get relative distance
        cam_id_max = self.controable_agent+1
        if 'Adv' in self.target:
            cam_id_max = 3
        states, self.obj_pos, depth_list = self.unrealcv.get_pose_img_batch(self.player_list, self.cam_id[1:cam_id_max],
                                                                    self.observation_type, 'bmp')
        self.obj_pos[0] = self.unrealcv.get_pose(self.cam_id[1])
        # for recording demo
        if self.get_bbox:
            mask = self.unrealcv.read_image(self.cam_id[1], 'object_mask', 'fast')
            mask, bbox = self.unrealcv.get_bbox(mask, self.player_list[1], normalize=False)
            self.bbox = bbox
            # im_disp = states[0][:, :, :3].copy()
            # cv2.rectangle(im_disp, (int(bbox[0]), int(bbox[1])), (int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])), (0, 255, 0), 5)
            # cv2.imshow('track_res', im_disp)
            # cv2.waitKey(1)

        states = np.array(states)
        if cam_id_max < self.controable_agent + 1:
            states = np.repeat(states, self.controable_agent, axis=0)

        pose_obs = []
        relative_pose = np.zeros((self.player_num, self.player_num, 2))
        # cauclate relative poses
        for j in range(self.player_num):
            vectors = []
            for i in range(self.player_num):
                obs, distance, direction = self.get_relative(self.obj_pos[j], self.obj_pos[i])
                yaw = self.obj_pos[j][4]/180*np.pi
                abs_loc = [self.obj_pos[i][0]/self.exp_distance, self.obj_pos[i][1]/self.exp_distance,
                           self.obj_pos[i][2]/self.exp_distance, np.cos(yaw), np.sin(yaw)]
                obs = obs + abs_loc
                vectors.append(obs)
                relative_pose[j, i] = np.array([distance, direction])
            pose_obs.append(vectors)
        info['Pose'] = self.obj_pos[0]
        info['Distance'], info['Direction'] = relative_pose[0][1]
        info['Relative_Pose'] = relative_pose
        self.pose_obs = np.array(pose_obs)
        info['Pose_Obs'] = self.pose_obs
        
        # set top_down camera
        if self.top:
            self.set_topview(info['Pose'], self.cam_id[0])

        info['Color'] = self.img_color = states[0][:, :, :3]

        metrics, score4tracker = self.relative_metrics(relative_pose)
        self.mis_lead = metrics['mislead']
        if 'distance' in self.reward_type:
            r_tracker = score4tracker[1] - metrics['collision'][0][1:].max()  # not clip for navigation
            rewards = []
            for i in range(len(self.player_list)):
                if i == 0:
                    rewards.append(r_tracker)
                elif i == 1:  # target, try to run away
                    r_target = - r_tracker - metrics['collision'][0][i]
                    rewards.append(r_target)
                else: # distractors, try to mislead tracker, and improve the target's reward.
                    if 'Share' in self.target:
                        r_d = r_target - metrics['collision'][0][i]
                    else:
                        r_d = r_target + score4tracker[i]  - metrics['collision'][0][i]
                    rewards.append(r_d)

            info['Reward'] = np.array(rewards)[:self.controable_agent]

        if r_tracker <= -0.99 or not metrics['target_viewed']:  # lost/mislead
            info['in_area'] = np.array([1])
        else:
            info['in_area'] = np.array([0])
        info['metrics'] = metrics
        info['d_in'] = metrics['d_in']

        if not metrics['target_viewed']:
            self.count_close += 1
        else:
            self.count_close = 0
            self.live_time = time.time()

        lost_time = time.time() - self.live_time
        if (self.early_stop and lost_time > 5) or self.count_steps > self.max_steps:
            info['Done'] = True

        return states, info['Reward'], info['Done'], info

    def reset(self, ):
        self.C_reward = 0
        self.count_close = 0
        self.pose_obs_his = []
        if 'PZR' in self.target:
            self.w_p = 1
        else:
            self.w_p = 0
        self.count_steps = 0
        np.random.seed()
        # stop move
        for i, obj in enumerate(self.player_list):
            self.unrealcv.set_move(obj, 0, 0)
            self.unrealcv.set_speed(obj, 0)

        # reset target location
        self.unrealcv.set_obj_location(self.player_list[1], random.sample(self.safe_start, 1)[0])
        if self.reset_type >= 1:
            for obj in self.player_list[1:]:
                if self.env_name == 'MPRoom':
                    map_id = [2, 3, 6, 7, 9]
                    spline = False
                    app_id = np.random.choice(map_id)
                else:
                    map_id = [1, 2, 3, 4]
                    spline = True
                    app_id = map_id[self.person_id % len(map_id)]
                    self.person_id += 1

                self.unrealcv.set_appearance(obj, app_id, spline)

        # target appearance
        if self.reset_type >= 2:
            if self.env_name == 'MPRoom':  # random target texture
                for obj in self.player_list[1:]:
                    self.unrealcv.random_player_texture(obj, self.textures_list, 3)

            self.unrealcv.random_lit(self.light_list)

        # texture
        if self.reset_type >= 3:
            self.unrealcv.random_texture(self.background_list, self.textures_list, 3)

        # obstacle
        if self.reset_type >= 4:
            self.unrealcv.clean_obstacles()
            self.unrealcv.random_obstacles(self.objects_list, self.textures_list,
                                           15, self.reset_area, self.start_area, True)
            
        # init target location and get expected tracker location
        res = []
        # sample a target pose from reset area
        if self.env_name == 'MPRoom':
            target_loc, _ = self.unrealcv.get_startpoint(reset_area=self.reset_area, exp_height=self.height) 
            self.unrealcv.set_obj_location(self.player_list[1], target_loc)
            time.sleep(0.5)
            target_pos = self.unrealcv.get_obj_pose(self.player_list[1])
            res = self.unrealcv.get_startpoint(target_pos, self.exp_distance, self.reset_area, self.height)
        
        # reset at fix point
        while len(res) == 0:
            target_pos = random.sample(self.safe_start, 1)[0]
            self.unrealcv.set_obj_location(self.player_list[1], target_pos)
            time.sleep(0.5)
            target_pos = self.unrealcv.get_obj_pose(self.player_list[1])
            res = self.unrealcv.get_startpoint(target_pos, self.exp_distance, self.reset_area, self.height)

        # set tracker location
        cam_pos_exp, yaw_exp = res
        self.unrealcv.set_obj_location(self.player_list[0], cam_pos_exp)
        time.sleep(0.5)
        self.rotate2exp(yaw_exp, self.player_list[0])
        
        # get tracker's pose
        tracker_pos = self.unrealcv.get_pose(self.cam_id[1])
        self.obj_pos = [tracker_pos, target_pos]

        # new obj
        # self.player_num is set by env.seed()
        while len(self.player_list) < self.player_num:
            name = 'target_C_{0}'.format(len(self.player_list)+1)
            if name in self.freeze_list:
                self.freeze_list.remove(name)
            else:
                self.unrealcv.new_obj('target_C', name, random.sample(self.safe_start, 1)[0])
            self.unrealcv.set_obj_color(name, np.random.randint(0, 255, 3))
            self.unrealcv.set_random(name, 0)
            self.player_list.append(name)
            self.unrealcv.set_interval(self.interval, name)
            self.cam_id.append(self.cam_id[-1]+1)
        while len(self.player_list) > self.player_num:
            name = self.player_list.pop()
            self.cam_id.pop()
            self.freeze_list.append(name)
            # self.unrealcv.destroy_obj(name)

        for i, obj in enumerate(self.player_list[2:]):
            # reset and get new pos
            res = self.unrealcv.get_startpoint(target_pos, np.random.randint(self.exp_distance*1.5, self.max_distance*2),
                                                                     self.reset_area, self.height, None)
            if len(res)==0:
                res = self.unrealcv.get_startpoint(reset_area=self.reset_area, exp_height=self.height)
            elif len(res) == 2:
                cam_pos_exp, yaw_exp = res
                self.unrealcv.set_obj_location(obj, cam_pos_exp)
                self.rotate2exp(yaw_exp, obj, 10)

        # cam on top of tracker
        center_pos = [(self.reset_area[0]+self.reset_area[1])/2, (self.reset_area[2]+self.reset_area[3])/2, 2000]
        self.set_topview(center_pos, self.cam_id[0])
        time.sleep(0.5)

        # set controllable agent number
        self.controable_agent = 1
        if 'Adv' in self.target or 'PZR' in self.target:
            self.controable_agent = self.player_num
            if 'Nav' in self.target or 'Ram' in self.target:
                self.controable_agent = 2

        # set view point
        height = 60
        pitch = - 5
        self.unrealcv.set_cam(self.player_list[0], [30, 0, height],
                              [0, pitch, 0])

        # get state
        for _ in range(2):
            states, self.obj_pos, depth_list = self.unrealcv.get_pose_img_batch(self.player_list, self.cam_id[1:self.controable_agent+1],
                                                                    self.observation_type, 'bmp')
            time.sleep(0.5)
        states = np.array(states)
        self.img_color = states[0][:, :, :3]
        # get pose state
        pose_obs = []
        for j in range(self.player_num):
            vectors = []
            for i in range(self.player_num):
                obs, distance, direction = self.get_relative(self.obj_pos[j], self.obj_pos[i])
                yaw = self.obj_pos[j][4]/180*np.pi
                abs_loc = [self.obj_pos[i][0]/self.exp_distance, self.obj_pos[i][1]/self.exp_distance,
                           self.obj_pos[i][2]/self.exp_distance, np.cos(yaw), np.sin(yaw)]
                obs = obs + abs_loc
                vectors.append(obs)
            pose_obs.append(vectors)
        self.pose_obs = np.array(pose_obs)
        self.count_freeze = [0 for i in range(self.player_num)]
        if 'Nav' in self.target or 'Ram' in self.target:
            for i in range(len(self.random_agents)):
                self.random_agents[i].reset()

        self.bbox_init = []
        mask = self.unrealcv.read_image(self.cam_id[1], 'object_mask', 'fast')
        mask, bbox = self.unrealcv.get_bbox(mask, self.player_list[1], normalize=False)
        self.mask_percent = mask.sum()/(255 * self.resolution[0] * self.resolution[1])
        self.bbox_init.append(bbox)

        self.pose = []
        self.act_smooth = [np.zeros(2) for i in range(self.controable_agent)]
        self.live_time = time.time()
        return states

    def close(self):
        self.unrealcv.client.disconnect()
        self.unreal.close()

    def render(self, mode='rgb_array', close=False):
        if close==True:
            self.unreal.close()
        return self.img_color

    def seed(self, seed=None):
        if seed is not None:
            self.player_num = seed % (self.max_player_num-2) + 2

    def set_action_factors(self, action_factor = np.array([np.random.uniform(0.8, 1.5), np.random.uniform(0.5, 1.2)]),
                           smooth_factor = 0.6):
        self.action_factor = action_factor
        self.smooth_factor = smooth_factor

    def set_random_height(self, random=True):
        self.random_height = random

    def get_start_area(self, safe_start, safe_range):
        start_area = [safe_start[0]-safe_range, safe_start[0]+safe_range,
                     safe_start[1]-safe_range, safe_start[1]+safe_range]
        return start_area

    def set_early_stop(self, do=True):
        self.early_stop = do

    def set_topview(self, current_pose, cam_id):
        cam_loc = current_pose[:3]
        cam_loc[-1] = current_pose[-1]+800
        cam_rot = [0, 0, -90]
        self.unrealcv.set_location(cam_id, cam_loc)
        self.unrealcv.set_rotation(cam_id, cam_rot)

    def get_relative(self, pose0, pose1):  # pose0-centric
        delt_yaw = pose1[4] - pose0[4]
        angle = misc.get_direction(pose0, pose1)
        distance = self.unrealcv.get_distance(pose1, pose0, 2)
        distance_norm = distance / self.exp_distance
        obs_vector = [np.sin(delt_yaw/180*np.pi), np.cos(delt_yaw/180*np.pi),
                      np.sin(angle/180*np.pi), np.cos(angle/180*np.pi),
                      distance_norm]
        return obs_vector, distance, angle

    def rotate2exp(self, yaw_exp, obj, th=1):
        yaw_pre = self.unrealcv.get_obj_rotation(obj)[1]
        delta_yaw = yaw_exp - yaw_pre
        while abs(delta_yaw) > th:
            self.unrealcv.set_move(obj, delta_yaw, 0)
            yaw_pre = self.unrealcv.get_obj_rotation(obj)[1]
            delta_yaw = (yaw_exp - yaw_pre) % 360
            if delta_yaw > 180:
                delta_yaw = 360 - delta_yaw
        return delta_yaw

    def relative_metrics(self, relative_pose):
        info = dict()
        relative_dis = relative_pose[:, :, 0]
        relative_ori = relative_pose[:, :, 1]
        collision_mat = np.zeros_like(relative_dis)
        collision_mat[np.where(relative_dis < 100)] = 1
        collision_mat[np.where(np.fabs(relative_ori) > 45)] = 0  # collision should be at the front view
        info['collision'] = collision_mat

        info['dis_ave'] = relative_dis.mean() # average distance among players, regard as a kind of density metric

        # if in the tracker's view
        view_mat = np.zeros_like(relative_ori)
        view_mat[np.where(np.fabs(relative_ori) < 45)] = 1
        view_mat[np.where(relative_dis > self.max_distance)] = 0
        view_mat_tracker = view_mat[0]
        # how many distractors are observed
        info['d_in'] = view_mat_tracker[2:].sum()
        info['target_viewed'] = view_mat_tracker[1] # target in the observable area

        relative_oir_norm = np.fabs(relative_ori) / 45.0
        relation_norm = np.fabs(relative_dis - self.exp_distance)/self.exp_distance + relative_oir_norm
        reward_tracker = 1 - relation_norm[0]  # measuring the quality among tracker to others
        info['tracked_id'] = np.argmax(reward_tracker)  # which one is tracked
        info['perfect'] = info['target_viewed'] * (info['d_in'] == 0) * (reward_tracker[1] > 0.5)
        info['mislead'] = 0
        if info['tracked_id'] > 1 and reward_tracker[info['tracked_id']] > 0.5: # only when target is far away to the center and distracotr is close
            advantage = reward_tracker[info['tracked_id']] - reward_tracker[1]
            if advantage > 1:
                info['mislead'] = info['tracked_id']

        return info, reward_tracker