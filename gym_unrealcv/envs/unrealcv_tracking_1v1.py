import time
import gym
import numpy as np
from gym import spaces
from gym_unrealcv.envs.tracking import reward, baseline
from gym_unrealcv.envs.utils import env_unreal, misc
from gym_unrealcv.envs.tracking.interaction import Tracking

''' 
It is an env for active object tracking.

State : raw color image and depth
Action:  (linear velocity ,angle velocity) 
Done : the relative distance or angle to target is larger than the threshold.
Task: Learn to follow the target object(moving person) in the scene
0: Tracker  1: Target
'''


class UnrealCvTracking_1v1(gym.Env):
    def __init__(self,
                 setting_file,
                 reset_type,
                 action_type='discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 reward_type='distance',  # distance
                 docker=False,
                 resolution=(160, 120),
                 target='Ram',  #Rule for target: Ram, Goal, Internal
                 ):
        self.docker = docker
        self.reset_type = reset_type
        self.roll = 0
        self.target = target
        setting = misc.load_env_setting(setting_file)
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
        self.safe_start = setting['safe_start']
        self.start_area = self.get_start_area(self.safe_start[0], 500)
        self.textures_list = misc.get_textures(setting['imgs_dir'], self.docker)

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
        self.action_space = [action_space, action_space_forward]

        # define observation space,
        # color, depth, rgbd,...
        self.observation_type = observation_type
        assert self.observation_type in ['Color', 'Depth', 'Rgbd', 'Gray']
        observation_space = self.unrealcv.define_observation(self.cam_id[0], self.observation_type, 'direct')
        self.observation_space = [observation_space, observation_space]

        self.unrealcv.pitch = self.pitch
        # define reward type
        self.reward_type = reward_type
        self.reward_function = reward.Reward(setting)

        if self.reset_type >= 5:
            self.unrealcv.init_objects(self.objects_env)

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
        if 'Ram' in self.target:
            self.random_agent = baseline.RandomAgent(action_space_forward)
        if 'Nav' in self.target:
            self.random_agent = baseline.GoalNavAgent(self.continous_actions_forward, self.reset_area, self.target)
        if 'Internal' in self.target:
            self.unrealcv.random_character(self.target_list[1])
        if self.target == 'Adv' or self.target == 'PZR':
            self.learn_target = True
        else:
            self.learn_target = False
        self.unrealcv.set_interval(setting['interval'])
        self.w_p = 1.0
        self.ep_lens = []

    def step(self, actions):
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
        self.count_steps += 1
        if self.action_type == 'Discrete':
            (velocity0, angle0) = self.discrete_actions[actions[0]]
            (velocity1, angle1) = self.discrete_actions[actions[1]]
        else:
            (velocity0, angle0) = actions[0]
            (velocity1, angle1) = actions[1]

        if 'Ram' in self.target:
            if self.action_type == 'Discrete':
                (velocity1, angle1) = self.discrete_actions[self.random_agent.act(self.target_pos)]
            else:
                (velocity1, angle1) = self.random_agent.act(self.target_pos)
        if 'Nav' in self.target:
            (velocity1, angle1) = self.random_agent.act(self.target_pos)

        self.unrealcv.set_move(self.target_list[0], angle0, velocity0)
        if 'Internal' not in self.target:
            self.unrealcv.set_move(self.target_list[1], angle1, velocity1)

        info['Pose'] = self.unrealcv.get_obj_pose(self.target_list[0])  # tracker pose
        target_pos = self.unrealcv.get_obj_pose(self.target_list[1])
        moved = np.linalg.norm(np.array(self.target_pos) - np.array(target_pos))
        self.target_pos = target_pos
        info['Direction'] = misc.get_direction(info['Pose'], self.target_pos)
        info['Distance'] = self.unrealcv.get_distance(self.target_pos, info['Pose'], 2)

        # update observation
        state_0 = self.unrealcv.get_observation(self.cam_id[0], self.observation_type, 'fast')
        if self.learn_target:
            state_1 = self.unrealcv.get_observation(self.cam_id[1], self.observation_type, 'fast')
        else:
            state_1 = state_0.copy()
        self.states = np.array([state_0, state_1])

        info['Color'] = self.unrealcv.img_color
        info['Depth'] = self.unrealcv.img_depth

        if (self.target=='Adv' or self.target=='PZR') and moved < 5:
            target_collision = True
        else:
            target_collision = False
        if 'distance' in self.reward_type:
            reward_0 = self.reward_function.reward_distance(info['Distance'], info['Direction'])
            reward_1 = self.reward_function.reward_target(info['Distance'], info['Direction'], None, self.w_p)
            reward_1 -= 0.5*target_collision
            info['Reward'] = np.array([reward_0, reward_1])

        if reward_0 <= -0.99 or target_collision or info['Collision']:
            self.count_close += 1
        else:
            self.count_close = 0

        if self.count_close > 20:
           info['Done'] = True
        # save the trajectory
        self.trajectory.append([info['Distance'], info['Direction']])
        info['Trajectory'] = self.trajectory

        return self.states, info['Reward'], info['Done'], info

    def reset(self, ):
        self.C_reward = 0
        self.count_close = 0
        self.count_eps += 1
        self.ep_lens.append(self.count_steps)

        # adaptive weight
        if 'PZR' in self.target:
            self.w_p = 1
        else:
            self.w_p = 0
        self.count_steps = 0
        # stop move
        self.unrealcv.set_move(self.target_list[0], 0, 0)
        self.unrealcv.set_move(self.target_list[1], 0, 0)
        np.random.seed()
        if 'Fix' in self.target:
            self.unrealcv.set_obj_location(self.target_list[1],
                                           [self.reset_area[0]/2, self.reset_area[2]/2, self.safe_start[0][-1]])
        else:
            self.unrealcv.set_obj_location(self.target_list[1], self.safe_start[0])

        # player shape
        if self.reset_type >= 1:
            if self.env_name == 'TrainRoom':
                map_id = [2, 3, 6, 7, 9]
                spline = False
                object_app = np.random.choice(map_id)
                tracker_app = np.random.choice(map_id)
            else:
                map_id = [1, 2, 3, 4]
                spline = True
                object_app = map_id[int(self.person_id % len(map_id))]
                tracker_app = map_id[int(self.person_id % len(map_id))]
                self.person_id += 1
            self.unrealcv.set_appearance(self.target_list[0], object_app, spline)
            self.unrealcv.set_appearance(self.target_list[1], tracker_app, spline)

        # player texture
        if self.reset_type >= 2:
            if self.env_name == 'TrainRoom':  # random target texture
                self.unrealcv.random_player_texture(self.target_list[0], self.textures_list, 3)
                self.unrealcv.random_player_texture(self.target_list[1], self.textures_list, 3)

        # light
        if self.reset_type >= 3:
            self.unrealcv.random_lit(self.light_list)

        # texture
        if self.reset_type >= 4:
            self.unrealcv.random_texture(self.background_list, self.textures_list, 3)

        # obstacle
        if self.reset_type >= 5:
            self.unrealcv.clean_obstacles()
            self.unrealcv.random_obstacles(self.objects_env, self.textures_list,
                                           20, self.reset_area, self.start_area)

        self.target_pos = self.unrealcv.get_obj_pose(self.target_list[1])

        res = self.unrealcv.get_startpoint(self.target_pos, self.exp_distance, self.reset_area,
                                           self.height, self.direction)

        count = 0
        while not res:
            count += 1
            time.sleep(0.1)
            self.target_pos = self.unrealcv.get_obj_pose(self.target_list[1])
            res = self.unrealcv.get_startpoint(self.target_pos, self.exp_distance, self.reset_area)
        cam_pos_exp, yaw = res
        cam_pos_exp[-1] = self.height
        self.unrealcv.set_obj_location(self.target_list[0], cam_pos_exp)
        yaw_pre = self.unrealcv.get_obj_rotation(self.target_list[0])[1]
        delta_yaw = yaw-yaw_pre
        while abs(delta_yaw) > 3:
            self.unrealcv.set_move(self.target_list[0], delta_yaw, 0)
            yaw_pre = self.unrealcv.get_obj_rotation(self.target_list[0])[1]
            delta_yaw = (yaw - yaw_pre) % 360
            if delta_yaw > 180:
                delta_yaw = 360 - delta_yaw
        current_pose = self.unrealcv.get_obj_pose(self.target_list[0])

        # get state
        state_0 = self.unrealcv.get_observation(self.cam_id[0], self.observation_type, 'fast')
        if self.learn_target:
            state_1 = self.unrealcv.get_observation(self.cam_id[1], self.observation_type, 'fast')
        else:
            state_1 = state_0.copy()
        self.states = np.array([state_0, state_1])

        # start target
        if 'Ram' in self.target or 'Nav' in self.target:
            self.random_agent.reset()
        if 'Internal' in self.target:
            self.unrealcv.random_character(self.target_list[1])

        self.trajectory = []
        self.trajectory.append(current_pose)
        return self.states

    def close(self):
        self.unreal.close()

    def render(self, mode='rgb_array', close=False):
        if close==True:
            self.unreal.close()
        return self.states[0]

    def seed(self, seed=None):
        self.person_id = seed

    def get_start_area(self, safe_start, safe_range):
        start_area = [safe_start[0]-safe_range, safe_start[0]+safe_range,
                      safe_start[1]-safe_range, safe_start[1]+safe_range]
        return start_area

