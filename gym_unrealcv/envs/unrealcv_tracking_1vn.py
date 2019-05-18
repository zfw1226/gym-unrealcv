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

# 0: tracker 1:target 2~n:others
# 0: fpv     1~n: top-view+relative location
# cam_id  0:global 1:tracker 2:target 3:others
class UnrealCvTracking_1vn(gym.Env):
    def __init__(self,
                 setting_file,
                 reset_type,
                 action_type='discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 reward_type='distance',  # distance
                 docker=False,
                 resolution=(80, 80),
                 nav='Random',  # Random, Goal, Internal
                 ):
        self.docker = docker
        self.reset_type = reset_type
        self.roll = 0
        self.nav = nav
        setting = self.load_env_setting(setting_file)
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
        elif self.action_type == 'Continuous':
            self.action_space = [spaces.Box(low=np.array(self.continous_actions['low']),
                                      high=np.array(self.continous_actions['high'])) for i in range(self.max_player_num)]
            player_action_space = spaces.Discrete(len(self.continous_actions_player))

        self.count_steps = 0

        # define observation space,
        # color, depth, rgbd,...
        self.observation_type = observation_type
        assert self.observation_type in ['Color', 'Depth', 'Rgbd', 'Gray', 'CG']
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

        self.person_id = 0
        if 'Random' in self.nav:
            self.random_agents = [RandomAgent(player_action_space) for i in range(self.max_player_num)]
        if 'Goal' in self.nav:
            self.random_agents = [GoalNavAgent(self.continous_actions_player, self.reset_area, self.nav) for i in range(self.max_player_num)]
        if 'Internal' in self.nav:
            self.unrealcv.set_random(self.player_list[0])
            self.unrealcv.set_maxdis2goal(target=self.player_list[0], dis=500)
        if 'Interval' in self.nav:
            self.unrealcv.set_interval(30)

        self.player_num = self.max_player_num

    def _step(self, actions):
        info = dict(
            Collision=False,
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
        actions = np.squeeze(actions)

        for i in range(len(self.player_list)):
            if i == 0 or ('Random' not in self.nav and 'Goal' not in self.nav):
                if self.action_type == 'Discrete':
                    actions2player.append(self.discrete_actions[actions[i]])
                else:
                    actions2player.append(actions[i])
            else:
                if 'Random' in self.nav:
                    if self.action_type == 'Discrete':
                        actions2player.append(self.discrete_actions_player[self.random_agents[i].act(self.obj_pos[i])])
                    else:
                        actions2player.append(self.random_agents[i].act(self.obj_pos[i]))
                if 'Goal' in self.nav:
                        actions2player.append(self.random_agents[i].act(self.obj_pos[i]))

        for i, player in enumerate(self.player_list):
            self.unrealcv.set_move(player, actions2player[i][1], actions2player[i][0])

        self.count_steps += 1

        # get relative distance
        pose_obs = []
        relative_pose = []
        for i, obj in enumerate(self.player_list):
            self.obj_pos[i] = self.unrealcv.get_obj_pose(obj)
        for j in range(self.player_num):
            vectors = []
            for i in range(self.player_num):
                obs, distance, direction = self.get_relative(self.obj_pos[j], self.obj_pos[i])
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
            self.set_topview(info['Pose'], self.cam_id[2])
        
        # update observation
        # state_0 = self.unrealcv.get_observation(self.cam_id[0], self.observation_type, 'fast')

        states = []
        for i in range(len(self.player_list)):
            if i == 0:
                state_img = self.unrealcv.get_observation(self.cam_id[0], self.observation_type, 'fast')
            elif i == 1:
                if self.top:
                    state_img = self.unrealcv.get_observation(self.cam_id[2], self.observation_type, 'fast')
                else:
                    state_img = self.unrealcv.get_observation(self.cam_id[1], self.observation_type, 'fast')
            else:
                if 'Random' in self.nav or 'Goal' in self.nav:
                    break
                if i > 1 and not self.top:
                    state_img = self.unrealcv.get_observation(self.cam_id[1]+i-1, self.observation_type, 'fast')
            states.append(state_img)
        states = np.array(states)
        # states = np.array([state_0, state_1])

        info['Color'] = self.unrealcv.img_color = states[0][:, :, :3]
        # print (info['Color'].shape)
        # cv2.imshow('tracker', states[0])
        # cv2.imshow('target', states[1])
        # cv2.imshow('t0', states[2])
        # cv2.imshow('t1', states[3])
        # cv2.waitKey(1)

        if 'distance' in self.reward_type:
            r_tracker = self.reward_function.reward_distance(info['Distance'], info['Direction'])
            r_target = self.reward_function.reward_target(info['Distance'], info['Direction'], None, self.w_p)
            rewards = []
            for i in range(len(self.player_list)):
                if i == 0:
                    rewards.append(r_tracker)
                elif i == 1:
                    rewards.append(r_target)
                else:
                    if 'Random' in self.nav or 'Goal' in self.nav:
                        break
                    r_d, mislead = self.reward_function.reward_distractor(relative_pose[i][0], relative_pose[i][1],
                                                                          self.player_num - 2)
                    rewards.append(r_d)
                    if mislead:
                        rewards[0] -= r_d
            info['Reward'] = np.array(rewards)

        if r_tracker <= -0.99 or info['Collision']:
            self.count_close += 1
        else:
            self.count_close = 0

        if self.count_close > 20 or self.count_steps > self.max_steps:
           info['Done'] = True

        return states, info['Reward'], info['Done'], info

    def _reset(self, ):
        self.C_reward = 0
        self.count_close = 0
        # self.count_eps += 1
        # self.ep_lens.append(self.count_steps)

        # adaptive weight
        if 'Dynamic' in self.nav:
            ep_lens_mean = np.array(self.ep_lens[-100:]).mean()
            self.w_p = 1 - int(ep_lens_mean/100)/5.0
        elif 'PZR' in self.nav:
            self.w_p = 1
        else:
            self.w_p = 0
        self.count_steps = 0
        # stop move
        for i, obj in enumerate(self.player_list):
            self.unrealcv.set_move(obj, 0, 0)
        np.random.seed()
        #  self.exp_distance = np.random.randint(150, 250)
        if 'Fix' in self.nav:
            self.unrealcv.set_obj_location(self.player_list[1], [self.reset_area[0]/2, self.reset_area[2]/2, self.safe_start[0][-1]])
        else:
            self.unrealcv.set_obj_location(self.player_list[1], self.safe_start[0])
        if self.reset_type >= 1:
            for i, obj in enumerate(self.player_list):
                if self.env_name == 'MPRoom':
                    #  map_id = [0, 2, 3, 7, 8, 9]
                    map_id = [2, 3, 6, 7, 9]
                    spline = False
                    app_id = np.random.choice(map_id)
                    tracker_app = np.random.choice(map_id)
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
                for i, obj in enumerate(self.player_list):
                    self.unrealcv.random_player_texture(obj, self.textures_list, 3)

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

        # texture
        if self.reset_type >= 3:
            self.unrealcv.random_texture(self.background_list, self.textures_list, 3)

        # moving objs
        # if self.reset_type >= 3: #TOOD
        #     self.unrealcv.new_obj(4, [0, 0, 0])

        # obstacle
        if self.reset_type >= 5:
            self.unrealcv.clean_obstacles()
            self.unrealcv.random_obstacles(self.objects_env, self.textures_list,
                                           20, self.reset_area, self.start_area)

        target_pos = self.unrealcv.get_obj_pose(self.player_list[1])
        # init tracker
        res = self.unrealcv.get_startpoint(target_pos, self.exp_distance, self.reset_area, self.height, self.direction)
        cam_pos_exp, yaw_exp = res
        self.unrealcv.set_obj_location(self.player_list[0], cam_pos_exp)
        self.rotate2exp(yaw_exp, self.player_list[0])

        # tracker's pose
        tracker_pos = self.unrealcv.get_obj_pose(self.player_list[0])
        self.obj_pos = [tracker_pos, target_pos]

        # new obj
        # self.player_num = np.random.randint(2, self.max_player_num)
        # self.player_num = self.max_player_num
        while len(self.player_list) < self.player_num:
            name = self.unrealcv.new_obj(4, self.safe_start[1])
            self.player_list.append(name)
        while len(self.player_list) > self.player_num:
            name = self.player_list.pop()
            self.unrealcv.destroy_obj(name)

        for i, obj in enumerate(self.player_list[2:]):
            # reset and get new pos
            cam_pos_exp, yaw_exp = self.unrealcv.get_startpoint(target_pos, self.max_distance, self.reset_area,
                                                                self.height, None)
            self.unrealcv.set_obj_location(obj, cam_pos_exp)
            self.rotate2exp(yaw_exp, obj, 30)
            self.obj_pos.append(self.unrealcv.get_obj_pose(obj))

        # cam on top of tracker

        self.set_topview(self.obj_pos[0], self.cam_id[2])

        # get state
        states = []
        for i in range(len(self.player_list)):
            if i == 0:
                state_img = self.unrealcv.get_observation(self.cam_id[0], self.observation_type, 'fast')
            elif i == 1:
                if self.top:
                    state_img = self.unrealcv.get_observation(self.cam_id[2], self.observation_type, 'fast')
                else:
                    state_img = self.unrealcv.get_observation(self.cam_id[1], self.observation_type, 'fast')
            else:
                if 'Random' in self.nav or 'Goal' in self.nav:
                    break
                if i > 1 and not self.top:
                    state_img = self.unrealcv.get_observation(self.cam_id[1]+i-1, self.observation_type, 'fast')
            states.append(state_img)

        states = np.array(states)
        # cv2.imshow('tracker', states[2])
        # cv2.imshow('target', state_1)
        # cv2.waitKey(1)
        # get pose state
        pose_obs = []
        for i, obj in enumerate(self.player_list):
            self.obj_pos[i] = self.unrealcv.get_obj_pose(obj)
        for j in range(self.player_num):
            vectors = []
            for i in range(self.player_num):
                obs, distance, direction = self.get_relative(self.obj_pos[j], self.obj_pos[i])
                vectors.append(obs)
            pose_obs.append(vectors)
        self.pose_obs = np.array(pose_obs)

        if 'Random' in self.nav or 'Goal' in self.nav:
            for i in range(len(self.random_agents)):
                self.random_agents[i].reset()
        if 'Internal' in self.nav:
            self.unrealcv.set_speed(self.player_list[1], np.random.randint(30, 200))
        self.pose = []
        return states

    def _close(self):
        self.unreal.close()

    def _render(self, mode='rgb_array', close=False):
        if close==True:
            self.unreal.close()
        return self.unrealcv.img_color/255.0

    def _seed(self, seed=None):
        if seed is not None:
            # self.person_id = seed
            self.player_num = seed % (self.max_player_num-2) + 2

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

    def set_topview(self, current_pose, cam_id):
        cam_loc = current_pose[:3]
        cam_loc[-1] = current_pose[-1]+800
        cam_rot = current_pose[-3:]
        cam_rot[-1] = -90
        self.unrealcv.set_location(cam_id, cam_loc)
        self.unrealcv.set_rotation(cam_id, cam_rot)

    def get_relative(self, pose0, pose1): # pose0-centric
        delt_yaw = pose1[4] - pose0[4]
        angle = self.get_direction(pose0, pose1)
        distance = self.unrealcv.get_distance(pose1, pose0, 2)
        distance_norm = distance / self.exp_distance
        obs_vector = [np.sin(delt_yaw/180*np.pi), np.cos(delt_yaw/180*np.pi),
                      np.sin(angle/180*np.pi), np.cos(angle/180*np.pi),
                      distance_norm]
        return obs_vector, distance, angle

    def rotate2exp(self, yaw_exp, obj, th=3):
        yaw_pre = self.unrealcv.get_obj_rotation(obj)[1]
        delta_yaw = yaw_exp - yaw_pre
        while abs(delta_yaw) > th:
            self.unrealcv.set_move(obj, delta_yaw, 0)
            yaw_pre = self.unrealcv.get_obj_rotation(obj)[1]
            delta_yaw = (yaw_exp - yaw_pre) % 360
            if delta_yaw > 180:
                delta_yaw = 360 - delta_yaw
        return delta_yaw

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.step_counter = 0
        self.keep_steps = 0
        self.action_space = action_space

    def act(self, pose):
        self.step_counter += 1
        if self.pose_last == None:
            self.pose_last = pose
            d_moved = 100
        else:
            d_moved = np.linalg.norm(np.array(self.pose_last) - np.array(pose))
            self.pose_last = pose
        if self.step_counter > self.keep_steps or d_moved < 3:
            self.action = self.action_space.sample()
            if self.action == 1 or self.action == 6 or self.action == 0:
                self.action = 0
                self.keep_steps = np.random.randint(10, 20)
            elif self.action == 2 or self.action == 3:
                self.keep_steps = np.random.randint(1, 20)
            else:
                self.keep_steps = np.random.randint(1, 10)
        return self.action

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0
        self.pose_last = None

class GoalNavAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space, goal_area, nav):
        self.step_counter = 0
        self.keep_steps = 0
        self.velocity_high = action_space['high'][0]
        self.velocity_low = action_space['low'][0]
        self.angle_high = action_space['high'][1]
        self.angle_low = action_space['low'][1]
        self.goal_area = goal_area
        # self.goal = self.generate_goal(self.goal_area)
        if 'Base' in nav:
            self.discrete = True
        else:
            self.discrete = False
        if 'Old' in nav:
            self.max_len = 30
        else:
            self.max_len = 1000
        if 'Fix' in nav:
            self.fix = True
        else:
            self.fix = False

    def act(self, pose):
        self.step_counter += 1
        if self.pose_last == None or self.fix:
            self.pose_last = pose
            d_moved = 100
        else:
            d_moved = np.linalg.norm(np.array(self.pose_last) - np.array(pose))
            self.pose_last = pose
        if self.check_reach(self.goal, pose) or d_moved < 3 or self.step_counter > self.max_len:
            self.goal = self.generate_goal(self.goal_area, self.fix)
            if self.discrete or self.fix:
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
            elif delt_yaw < -3:
                self.angle = self.angle_low / 2
        else:
            self.angle = np.clip(delt_yaw, self.angle_low, self.angle_high)
            # self.angle = delt_yaw
            velocity = self.velocity * (1 + 0.2*np.random.random())
        return (velocity, self.angle)

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0
        self.goal_id = 0
        self.goal = self.generate_goal(self.goal_area, self.fix)
        self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
        self.pose_last = None

    def generate_goal(self, goal_area, fixed=False):
        goal_list = [[goal_area[0], goal_area[2]], [goal_area[0], goal_area[3]],
                     [goal_area[1], goal_area[3]], [goal_area[1], goal_area[2]]]

        if fixed:
            goal = np.array(goal_list[self.goal_id%len(goal_list)])/2
            self.goal_id += 1

        else:
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