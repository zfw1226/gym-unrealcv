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
                 resolution=(160, 120),
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
        self.objects_env = setting['objects_list']
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
        # define reward type
        # distance, bbox, bbox_distance,
        self.reward_type = reward_type
        self.reward_function = reward.Reward(setting)

        self.rendering = False

        if self.reset_type >= 4:
            self.unrealcv.init_objects(self.objects_env)

        self.count_close = 0

        if self.reset_type == 5:
            self.unrealcv.simulate_physics(self.objects_env)

        self.unrealcv.set_random(self.player_list[0], 0)
        self.unrealcv.set_random(self.player_list[1], 0)

        self.person_id = 0
        if 'Ram' in self.target:
            self.random_agents = [baseline.RandomAgent(player_action_space) for i in range(self.max_player_num)]
        elif 'Nav' in self.target:
            if 'Goal' in self.target:
                th = 0.5
            else:
                th = 0
            self.random_agents = [baseline.GoalNavAgent(self.continous_actions_player, self.reset_area, self.target, th
                                                        ) for i in range(self.max_player_num)]

        for player in self.player_list:
            self.unrealcv.set_interval(self.interval, player)

        self.player_num = self.max_player_num

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
                    # fix camera
                    # actions[0] = 6
                    # add noise on movement
                    # actions2player.append(self.discrete_actions[actions[i]]*np.random.uniform(0.5, 1.5, 2))
                    actions2player.append(self.discrete_actions[actions[i]])
                else:
                    actions2player.append(actions[i])
                if self.count_freeze[i] > 5 and 'Fix' in self.target:
                    actions2player[i] = self.discrete_actions_player[-1]
            else:
                if 'Ram' in self.target:
                    if self.action_type == 'Discrete':
                        actions2player.append(self.discrete_actions_player[self.random_agents[i].act(self.obj_pos[i])])
                    else:
                        actions2player.append(self.random_agents[i].act(self.obj_pos[i]))
                        # self.discrete_actions[self.action_space[i].sample()]
                if 'Nav' in self.target:
                    if i == 1:
                        actions2player.append(self.random_agents[i].act(self.obj_pos[i]))
                    else:
                        actions2player.append(self.random_agents[i].act(self.obj_pos[i], self.random_agents[1].goal))

        self.unrealcv.set_move_batch(self.player_list, actions2player)
        self.count_steps += 1

        # get relative distance
        pose_obs = []
        relative_pose = []


        cam_id_max = self.controable_agent+1
        if 'Adv' in self.target:
            cam_id_max = 2
        states, self.obj_pos = self.unrealcv.get_pose_img_batch(self.player_list, self.cam_id[1:cam_id_max],
                                                                    self.observation_type, 'bmp')
        self.obj_pos[0] = self.unrealcv.get_pose(self.cam_id[1])
        states = np.array(states)
        if cam_id_max < self.controable_agent + 1:
            states = np.repeat(states, self.controable_agent, axis=0)

        for j in range(self.controable_agent):
            vectors = []
            for i in range(self.player_num):
                obs, distance, direction = self.get_relative(self.obj_pos[j], self.obj_pos[i])
                yaw = self.obj_pos[j][4]/180*np.pi
                abs_loc = [self.obj_pos[i][0]/self.exp_distance, self.obj_pos[i][1]/self.exp_distance,
                           self.obj_pos[i][2]/self.exp_distance, np.cos(yaw), np.sin(yaw)]
                obs = obs + abs_loc
                vectors.append(obs)
                if j==0:
                    relative_pose.append([distance, direction])
            pose_obs.append(vectors)
        info['Pose'] = self.obj_pos[0]
        info['Direction'] = relative_pose[1][1]
        info['Distance'] = relative_pose[1][0]
        info['Relative_Pose'] = relative_pose
        self.pose_obs = np.array(pose_obs)
        info['Pose_Obs'] = self.pose_obs
        # set top_down camera
        if self.top:
            self.set_topview(info['Pose'], self.cam_id[0])

        info['Color'] = self.unrealcv.img_color = states[0][:, :, :3]
        # cv2.imshow('tracker', states[0])
        # cv2.imshow('target', states[1])
        # cv2.waitKey(1)
        info['d_in'] = 0
        self.mis_lead = [0]
        reset_id = []
        if 'distance' in self.reward_type:
            r_tracker = self.reward_function.reward_distance(info['Distance'], info['Direction'])
            rewards = []
            for i in range(len(self.player_list)):
                if i == 0:
                    rewards.append(r_tracker)
                elif i == 1:
                    r_target = self.reward_function.reward_target(info['Distance'], info['Direction'], None, self.w_p)
                    rewards.append(r_target)
                else:
                    r_d, mislead, r_distract, observed, collision = self.reward_function.reward_distractor(relative_pose[i][0], relative_pose[i][1],
                                                                          self.player_num - 2)
                    info['d_in'] += observed
                    if mislead > 0:
                        rewards[0] -= r_distract
                        rewards[1] += r_distract
                        rewards[0] = max(rewards[0], -1)
                        rewards[1] = min(rewards[1], 1)
                    rewards.append(r_d)
                    if 'Nav' in self.target:
                        continue
                    info['Collision'] = max(collision, info['Collision'])
                    if collision == 1:
                         rewards[0] = -1
                    if relative_pose[i][0] > max(info['Distance'], self.exp_distance)*2:
                        reset_id.append(self.player_list[i])
                        self.count_freeze[i] = 0
                    if mislead == 1 or collision == 1:
                        self.count_freeze[i] = min(self.count_freeze[i] + 1, 10)

                    self.mis_lead.append(mislead)
            info['Reward'] = np.array(rewards)[:self.controable_agent]
        target_inarea = self.reward_function.target_inarea()
        if r_tracker <= -0.99 or max(self.mis_lead) >= 2 or not target_inarea:  # lost/mislead
            info['in_area'] = np.array([1])
        else:
            info['in_area'] = np.array([0])
        if info['d_in'] == 0 and max(self.mis_lead) == 0 and self.reward_function.target_incenter():
            info['perfect'] = 1
        else:
            info['perfect'] = 0
        '''
        if r_tracker > 0.5:
            cv2.imshow('good', states[0])
        if r_tracker < -0.5:
            cv2.imshow('bad', states[0])
        cv2.waitKey(1)
        '''
        if not target_inarea or max(self.mis_lead) >= 2:
            self.count_close += 1
        else:
            self.count_close = 0
            self.live_time = time.time()

        lost_time = time.time() - self.live_time
        if (self.count_close > 20 and lost_time > 5) or self.count_steps > self.max_steps:
            info['Done'] = True
        if 'Res' in self.target:
            for obj in reset_id:
                min_dis = max(info['Distance'], self.exp_distance)*1.1
                start_distance = np.random.randint(min(min_dis, self.max_direction*0.9), self.max_distance)
                res = self.unrealcv.get_startpoint(self.obj_pos[1], start_distance, self.reset_area, self.height)
                if len(res) == 2:
                    cam_pos_exp, yaw_exp = res
                    self.unrealcv.set_obj_location(obj, cam_pos_exp)
                    self.rotate2exp(yaw_exp, obj, 10)
                else:
                    info['Done'] = True
        return states, info['Reward'], info['Done'], info

    def reset(self, ):
        self.C_reward = 0
        self.count_close = 0
        if 'PZR' in self.target:
            self.w_p = 1
        else:
            self.w_p = 0
        self.count_steps = 0
        # stop move
        for i, obj in enumerate(self.player_list):
            self.unrealcv.set_move(obj, 0, 0)
            self.unrealcv.set_speed(obj, 0)
        np.random.seed()

        # reset target location
        self.unrealcv.set_obj_location(self.player_list[1], self.safe_start[0])
        if self.reset_type >= 1:
            for obj in self.player_list[1:]:
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
            self.unrealcv.random_obstacles(self.objects_env, self.textures_list,
                                           20, self.reset_area, self.start_area)

        target_pos = self.unrealcv.get_obj_pose(self.player_list[1])
        # init tracker
        res = self.unrealcv.get_startpoint(target_pos, self.exp_distance, self.reset_area, self.height)
        cam_pos_exp, yaw_exp = res
        self.unrealcv.set_obj_location(self.player_list[0], cam_pos_exp)
        time.sleep(0.5)
        self.rotate2exp(yaw_exp, self.player_list[0])
        # tracker's pose
        # tracker_pos = self.unrealcv.get_obj_pose(self.player_list[0])
        tracker_pos = self.unrealcv.get_pose(self.cam_id[1])
        self.obj_pos = [tracker_pos, target_pos]

        # new obj
        # self.player_num is set by env.seed()
        while len(self.player_list) < self.player_num:
            name = self.unrealcv.new_obj('target_C', 'target_C_{0}'.format(len(self.player_list)), self.safe_start[1])
            self.unrealcv.set_obj_color(name, np.random.randint(0, 255, 3))
            self.unrealcv.set_random(name, 0)
            self.player_list.append(name)
            self.unrealcv.set_interval(self.interval, name)
            self.cam_id.append(self.cam_id[-1]+1)
        while len(self.player_list) > self.player_num:
            name = self.player_list.pop()
            self.cam_id.pop()
            self.unrealcv.destroy_obj(name)

        for i, obj in enumerate(self.player_list[2:]):
            # reset and get new pos
            if 'Far' in self.target:
                res = self.unrealcv.get_startpoint(target_pos, None, self.reset_area, self.height, None)
            else:
                res = self.unrealcv.get_startpoint(target_pos, np.random.randint(self.exp_distance*1.1, self.max_distance*1.5),
                                                                     self.reset_area, self.height, None)
            if len(res) == 2:
                cam_pos_exp, yaw_exp = res
                self.unrealcv.set_obj_location(obj, cam_pos_exp)
                self.rotate2exp(yaw_exp, obj, 10)

        # cam on top of tracker

        self.set_topview(self.obj_pos[0], self.cam_id[0])
        time.sleep(0.5)

        # set controllable agent number
        self.controable_agent = 1
        if 'Adv' in self.target or 'PZR' in self.target:
            self.controable_agent = self.player_num
            if 'Nav' in self.target or 'Ram' in self.target:
                self.controable_agent = 2

        # randomize view point
        height = np.random.randint(-40, 80)
        pitch = -0.1 * height + np.random.randint(-5, 5)
        self.unrealcv.set_cam(self.player_list[0], [40, 0, height],
                              [np.random.randint(-3, 3), pitch, 0])
        # get state
        states, self.obj_pos = self.unrealcv.get_pose_img_batch(self.player_list, self.cam_id[1:self.controable_agent+1],
                                                                self.observation_type, 'bmp')
        states = np.array(states)
        self.unrealcv.img_color = states[0][:, :, :3]
        # get pose state
        pose_obs = []
        for j in range(self.controable_agent):
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
        self.pose = []
        self.live_time = time.time()
        return states

    def close(self):
        self.unreal.close()

    def render(self, mode='rgb_array', close=False):
        if close==True:
            self.unreal.close()
        return self.unrealcv.img_color

    def seed(self, seed=None):
        if seed is not None:
            self.player_num = seed % (self.max_player_num-2) + 2

    def get_start_area(self, safe_start, safe_range):
        start_area = [safe_start[0]-safe_range, safe_start[0]+safe_range,
                     safe_start[1]-safe_range, safe_start[1]+safe_range]
        return start_area

    def set_topview(self, current_pose, cam_id):
        cam_loc = current_pose[:3]
        cam_loc[-1] = current_pose[-1]+800
        cam_rot = current_pose[-3:]
        cam_rot[-1] = -90
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