import os
import time
import gym
import numpy as np
from gym import spaces
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
class UnrealCvTracking_general(gym.Env):
    def __init__(self,
                 setting_file,
                 reset_type=0,
                 action_type='Discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 reward_type='distance',  # distance
                 docker=False,
                 resolution=(480, 480)
                 ):
        self.docker = docker
        self.reset_type = reset_type
        setting = misc.load_env_setting(setting_file)
        self.env_name = setting['env_name']
        self.max_steps = setting['max_steps']
        self.height = setting['height']
        self.cam_id = [setting['third_cam']['cam_id']]
        self.agent_configs = setting['agents']
        self.env_configs = setting["env"]
        self.agents = misc.convert_dict(self.agent_configs)
        self.tracker_id = 0
        self.target_id = 1
        # print(self.agents)
        self.player_list = []
        self.freeze_list = []
        self.reward_config = setting['rewards']
        self.max_distance = self.reward_config['max_distance']
        self.max_distance = self.reward_config['max_distance']
        self.min_distance = self.reward_config['min_distance']
        self.max_direction = self.reward_config['max_direction']

        self.height_top_view = setting['third_cam']['height_top_view']
        # self.pitch = setting['pitch']
        self.objects_list = self.env_configs["objects"]
        self.reset_area = setting['reset_area']
        # self.background_list = setting['backgrounds']
        # self.light_list = setting['lights']
        self.max_player_num = setting['max_player_num']  # the max players number
        self.exp_distance = self.reward_config['exp_distance']
        texture_dir = setting['imgs_dir']
        gym_path = os.path.dirname(gym_unrealcv.__file__)
        texture_dir = os.path.join(gym_path, 'envs', 'UnrealEnv', texture_dir)
        self.textures_list = os.listdir(texture_dir)
        self.safe_start = setting['safe_start']
        self.interval = setting['interval']
        self.start_area = self.get_start_area(self.safe_start[0], 500) # the start area of the agent, where we don't put obstacles

        self.count_eps = 0
        self.count_steps = 0
        self.count_close = 0
        self.tracker_id = 0

        self.resolution = resolution
        self.display = None
        self.use_opengl = False
        self.offscreen_rendering = False
        self.nullrhi = False
        self.launched = False

        # define reward type: distance
        self.reward_type = reward_type

        for i in range(len(self.textures_list)):
            if self.docker:
                self.textures_list[i] = os.path.join('/unreal', setting['imgs_dir'], self.textures_list[i])
            else:
                self.textures_list[i] = os.path.join(texture_dir, self.textures_list[i])

        # init agents
        self.player_list = list(self.agents.keys())
        self.cam_list = [self.agents[player]['cam_id'] for player in self.player_list]

        # define action space
        self.action_type = action_type
        assert self.action_type in ['Discrete', 'Continuous']
        self.action_space = [self.define_action_space(self.action_type, self.agents[obj]) for obj in self.player_list]

        # define observation space,
        # color, depth, rgbd,...
        self.observation_type = observation_type
        assert self.observation_type in ['Color', 'Depth', 'Rgbd', 'Gray', 'CG', 'Mask', 'Pose']
        self.observation_space = [self.define_observation_space(self.cam_list[i], self.observation_type, resolution)
                                  for i in range(len(self.player_list))]

        # config unreal env
        if 'linux' in sys.platform:
            env_bin = setting['env_bin']
        elif 'win' in sys.platform:
            env_bin = setting['env_bin_win']
        if 'env_map' in setting.keys():
            env_map = setting['env_map']
        else:
            env_map = None
        self.unreal = env_unreal.RunUnreal(ENV_BIN=env_bin, ENV_MAP=env_map)

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
        for i, obj in enumerate(self.player_list):
            if actions[i] is None:  # if the action is None, then we don't control this agent
                actions2player.append(None)  # place holder
                continue
            if self.action_type == 'Discrete':
                act_index = actions[i]
                act_now = self.agents[obj]["discrete_action"][act_index]
                actions2player.append(act_now)
            else:
                actions2player.append(actions[i])

        for i, obj in enumerate(self.player_list):
            if actions2player[i] is None:
                continue
            self.unrealcv.set_move_new(obj, actions2player[i])
        # self.unrealcv.set_move_batch(self.player_list, actions2player)
        self.count_steps += 1

        # get states
        obj_poses, cam_poses, imgs, masks, depths = self.unrealcv.get_pose_img_batch(self.player_list, self.cam_list, self.cam_flag)
        self.obj_poses = obj_poses
        observations = self.prepare_observation(self.observation_type, imgs, masks, depths, obj_poses)

        self.img_show = self.prepare_img2show(self.tracker_id, observations)

        pose_obs, relative_pose = self.get_pose_states(obj_poses)
        metrics, score4tracker = self.relative_metrics(relative_pose, self.tracker_id, self.target_id)

        # prepare the info
        info['Pose'] = obj_poses[self.tracker_id]
        info['Distance'], info['Direction'] = relative_pose[self.tracker_id][self.target_id]
        info['Relative_Pose'] = relative_pose
        info['Pose_Obs'] = pose_obs
        info['Reward'] = self.get_rewards(score4tracker, metrics, self.tracker_id, self.target_id)
        info['metrics'] = metrics
        info['d_in'] = metrics['d_in']  # the number of distractor in tracker's view

        # meta info for evaluation
        if info['Reward'][self.tracker_id] <= -0.99 or not metrics['target_viewed']:  # lost/mislead
            info['in_area'] = np.array([1])
        else:
            info['in_area'] = np.array([0])

        if self.count_steps > self.max_steps:
            info['Done'] = True
        return observations, info['Reward'], info['Done'], info

    def reset(self, ):
        if not self.launched:  # first time to launch
            self.launched = self.launch_ue_env()
            self.init_agents()
            self.init_objects()

        self.count_close = 0
        self.count_steps = 0
        self.count_eps += 1

        # stop move and disable physics
        for i, obj in enumerate(self.player_list):
            if self.agents[obj]['agent_type'] in ['player', 'car', 'animal']:
                if not self.agents[obj]['internal_nav'] and i !=self.tracker_id:
                    self.unrealcv.set_move_new(obj, [0, 0])
                    self.unrealcv.set_phy(obj, 0)
                # self.unrealcv.set_speed(obj, 0)
            elif self.agents[obj]['agent_type'] == 'drone':
                self.unrealcv.set_move_new(obj, [0, 0, 0, 0])
                self.unrealcv.set_phy(obj, 0)

        # reset target location
        self.unrealcv.set_obj_location(self.player_list[self.target_id], random.sample(self.safe_start, 1)[0])

        if self.reset_type == 1:
            self.environment_augmentation(player_mesh=True, player_texture=False, light=False, background_texture=False,
                                          layout=False, layout_texture=False)
        elif self.reset_type == 2:
            self.environment_augmentation(player_mesh=True, player_texture=True, light=True, background_texture=False,
                                          layout=False, layout_texture=False)
        elif self.reset_type == 3:
            self.environment_augmentation(player_mesh=True, player_texture=True, light=True, background_texture=True,
                                          layout=False, layout_texture=False)
        elif self.reset_type == 4:
            self.environment_augmentation(player_mesh=True, player_texture=True, light=True, background_texture=False,
                                          layout=True, layout_texture=True)
        elif self.reset_type == 5:
            self.environment_augmentation(player_mesh=True, player_texture=True, light=True, background_texture=True,
                                          layout=True, layout_texture=True)
        # self.num_targets = self.num_agenets/2
        # self.target_list = random.sample(self.player_list, self.num_targets)

        # self.tracker_list = self.player_list.sample(self.num_tracker)
        # for i, obj in enumerate(self.target_list):
            # init target location and get expected tracker location
        target_pos, tracker_pos_exp = self.sample_target_init_pose(False)

        # set tracker location
        cam_pos_exp, yaw_exp = tracker_pos_exp
        tracker_name = self.player_list[self.tracker_id]
        self.unrealcv.set_obj_location(tracker_name, cam_pos_exp)
        # self.set_yaw(tracker_name, yaw_exp)
        self.rotate2exp(yaw_exp, self.player_list[self.tracker_id], 3)

        # get tracker's camera pose
        tracker_pos = self.unrealcv.get_pose(self.cam_list[self.tracker_id])

        for i, obj in enumerate(self.player_list):
            # reset and get new pos
            if i != self.tracker_id and i != self.target_id: # distractor
                res = self.unrealcv.get_startpoint(target_pos, np.random.randint(self.exp_distance*2, self.max_distance*3),
                                                                         self.reset_area, self.height, None)
                if len(res)==0:
                    res = self.unrealcv.get_startpoint(reset_area=self.reset_area, exp_height=self.height)
                cam_pos_exp, yaw_exp = res
                self.unrealcv.set_obj_location(obj, cam_pos_exp)
                # self.set_yaw(obj, yaw_exp)
                # self.rotate2exp(yaw_exp, obj, 10)

        #remove object between tracker and target
        for obj in self.objects_list:
            obj_loc = self.unrealcv.get_obj_location(obj)
            tracker_loc = self.unrealcv.get_obj_location(self.player_list[self.tracker_id])
            target_loc = self.unrealcv.get_obj_location(self.player_list[self.target_id])
            mid_point = [(tracker_loc[i] + target_loc[i]) / 2 for i in range(len(tracker_loc))]
            dis2obj = np.sqrt(np.square(obj_loc[0] - mid_point[0]) + np.square(obj_loc[1] - mid_point[1]))
            if dis2obj < 300:
                self.unrealcv.set_obj_location(obj, self.unrealcv.objects_dict[obj])
        # set view point
        for obj in self.player_list:
            self.unrealcv.set_cam(obj, self.agents[obj]['relative_location'], self.agents[obj]['relative_rotation'])
            self.unrealcv.set_phy(obj, 1) # enable physics

        # get state
        obj_poses, cam_poses, imgs, masks, depths = self.unrealcv.get_pose_img_batch(self.player_list, self.cam_list, self.cam_flag)
        observations = self.prepare_observation(self.observation_type, imgs, masks, depths, obj_poses)
        self.obj_poses = obj_poses
        self.img_show = self.prepare_img2show(self.tracker_id, observations)

        # get pose state
        # pose_obs, relative_pose = self.get_pose_states(obj_pos)

        return observations

    def close(self):
        self.unrealcv.client.disconnect()
        self.unreal.close()

    def render(self, mode='rgb_array', close=False):
        if close==True:
            self.unreal.close()
        return self.img_show

    def seed(self, seed=None):
        np.random.seed(seed)
        # if seed is not None:
        #     self.player_num = seed % (self.max_player_num-2) + 2

    def get_start_area(self, safe_start, safe_range):
        start_area = [safe_start[0]-safe_range, safe_start[0]+safe_range,
                     safe_start[1]-safe_range, safe_start[1]+safe_range]
        return start_area

    def set_topview(self, current_pose, cam_id):
        cam_loc = current_pose[:3]
        cam_loc[-1] = self.height_top_view
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

    def prepare_observation(self, observation_type, img_list, mask_list, depth_list, pose_list):
        if observation_type == 'Depth':
            return np.array(depth_list)
        elif observation_type == 'Mask':
            return np.array(mask_list)
        elif observation_type == 'Color':
            return np.array(img_list)
        elif observation_type == 'Rgbd':
            return np.append(np.array(img_list), np.array(depth_list), axis=-1)
        elif observation_type == 'Pose':
            return np.array(pose_list)

    def rotate2exp(self, yaw_exp, obj, th=1):
        yaw_pre = self.unrealcv.get_obj_rotation(obj)[1]
        delta_yaw = yaw_exp - yaw_pre
        while abs(delta_yaw) > th:
            if 'Drone' in obj:
                self.unrealcv.set_move_new(obj, [0, 0, 0, np.clip(delta_yaw, -60, 60)/60*np.pi])
            else:
                self.unrealcv.set_move_new(obj, [np.clip(delta_yaw, -60, 60), 0])
            yaw_pre = self.unrealcv.get_obj_rotation(obj)[1]
            delta_yaw = (yaw_exp - yaw_pre) % 360
            if delta_yaw > 180:
                delta_yaw = 360 - delta_yaw
        return delta_yaw

    def relative_metrics(self, relative_pose, tracker_id, target_id):
        # compute the relative relation (collision, in-the-view, misleading) among agents for rewards and evaluation metrics
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
        view_mat_tracker = view_mat[tracker_id]
        # how many distractors are observed
        info['d_in'] = view_mat_tracker.sum() - view_mat_tracker[target_id] - view_mat_tracker[tracker_id]  # distractor in the observable area
        info['target_viewed'] = view_mat_tracker[target_id]  # target in the observable area

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

    def get_rewards(self, score4tracker, metrics, tracker_id, target_id):
        rewards = []
        mask = np.ones(metrics['collision'][tracker_id].shape, dtype=bool)
        mask[tracker_id] = False
        tracker_collision = metrics['collision'][tracker_id]
        r_tracker = score4tracker[target_id] - np.max(tracker_collision[mask])  #
        r_target = -score4tracker[target_id]
        for i in range(len(self.player_list)):
            if i == tracker_id:
                rewards.append(r_tracker)
            elif i == target_id:  # target, try to run away
                rewards.append(r_target - tracker_collision[i])
            else:  # distractors, try to mislead tracker, and improve the target's reward.
                r_d = r_target + score4tracker[i]  # try to appear in the tracker's view
                r_d -= tracker_collision[i]
                rewards.append(r_d)
        return np.array(rewards)

    def add_agent(self, name, loc, refer_agent):
        # print(f'add {name}')
        new_dict = refer_agent.copy()
        cam_num = self.unrealcv.get_camera_num()
        self.unrealcv.new_obj(refer_agent['class_name'], name, random.sample(self.safe_start, 1)[0])
        self.player_list.append(name)
        if self.unrealcv.get_camera_num() > cam_num:
            new_dict['cam_id'] = cam_num
        else:
            new_dict['cam_id'] = -1
        self.cam_list.append(new_dict['cam_id'])
        self.unrealcv.set_obj_scale(name, refer_agent['scale'])
        self.unrealcv.set_obj_color(name, np.random.randint(0, 255, 3))
        self.unrealcv.set_random(name, 0)
        self.unrealcv.set_interval(self.interval, name)
        self.unrealcv.set_obj_location(name, loc)
        self.action_space.append(self.define_action_space(self.action_type, agent_info=new_dict))
        self.observation_space.append(self.define_observation_space(new_dict['cam_id'], self.observation_type, self.resolution))
        return new_dict

    def remove_agent(self, name, freeze=False):
        # print(f'remove {name}')
        agent_index = self.player_list.index(name)
        self.player_list.remove(name)
        self.cam_list = self.remove_cam(name)
        self.action_space.pop(agent_index)
        self.observation_space.pop(agent_index)
        if freeze:
            self.freeze_list.append(name)  # the agent still exists in the scene, but it is frozen
        else:
            self.unrealcv.destroy_obj(name)  # the agent is removed from the scene
            self.agents.pop(name)

    def remove_cam(self, name):
        cam_id = self.agents[name]['cam_id']
        cam_list = []
        for obj in self.player_list:
            if self.agents[obj]['cam_id'] > cam_id and cam_id > 0:
                self.agents[obj]['cam_id'] -= 1
            cam_list.append(self.agents[obj]['cam_id'])
        return cam_list

    def define_action_space(self, action_type, agent_info):
        if action_type == 'Discrete':
            return spaces.Discrete(len(agent_info["discrete_action"]))
        elif action_type == 'Continuous':
            return spaces.Box(low=np.array(agent_info["continuous_action"]['low']),
                              high=np.array(agent_info["continuous_action"]['high']), dtype=np.float32)

    def define_observation_space(self, cam_id, observation_type, resolution=(160, 120)):
        if observation_type == 'Pose' or cam_id < 0:
            observation_space = spaces.Box(low=-100, high=100, shape=(6,),
                                               dtype=np.float16)  # TODO check the range and shape
        else:
            if observation_type == 'Color' or observation_type == 'CG' or observation_type == 'Mask':
                img_shape = (resolution[1], resolution[0], 3)
                observation_space = spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
            elif observation_type == 'Depth':
                img_shape = (resolution[1], resolution[0], 1)
                observation_space = spaces.Box(low=0, high=100, shape=img_shape, dtype=np.float16)
            elif observation_type == 'Rgbd':
                s_low = np.zeros((resolution[1], resolution[0], 4))
                s_high = np.ones((resolution[1], resolution[0], 4))
                s_high[:, :, -1] = 100.0  # max_depth
                s_high[:, :, :-1] = 255  # max_rgb
                observation_space = spaces.Box(low=s_low, high=s_high, dtype=np.float16)

        return observation_space

    def sample_target_init_pose(self, use_reset_area=False):
        # sample a target pose
        tracker_pos_exp = []
        if use_reset_area:
            target_loc, _ = self.unrealcv.get_startpoint(reset_area=self.reset_area, exp_height=self.height)
            self.unrealcv.set_obj_location(self.player_list[self.target_id], target_loc)
            time.sleep(0.5)
            target_pos = self.unrealcv.get_obj_pose(self.player_list[self.target_id])
            tracker_pos_exp = self.unrealcv.get_startpoint(target_pos, self.exp_distance, self.reset_area, self.height) # get the expected tracker location

        # reset at fix point
        while len(tracker_pos_exp) == 0:
            target_pos = random.choice(self.safe_start) # sample one pre-defined start point
            self.unrealcv.set_obj_location(self.player_list[self.target_id], target_pos)
            time.sleep(0.5)
            target_pos = self.unrealcv.get_obj_pose(self.player_list[self.target_id])
            tracker_pos_exp = self.unrealcv.get_startpoint(target_pos, self.exp_distance, self.reset_area, self.height) # get the expected tracker location
        return target_pos, tracker_pos_exp

    def set_yaw(self, name, yaw_exp):
        if self.agents[name]['agent_type'] == 'car' or self.agents[name]['agent_type'] == 'drone':
            self.unrealcv.set_obj_rotation_bp(name, [0, yaw_exp, 0])
            time.sleep(0.3)
        else:
            self.unrealcv.set_obj_rotation(name, [0, yaw_exp, 0])


    def environment_augmentation(self, player_mesh=False, player_texture=False,
                                 light=False, background_texture=False,
                                 layout=False, layout_texture=False):
        if player_mesh:  # random human mesh
            for obj in self.player_list:
                if self.agents[obj]['agent_type'] == 'player':
                    if self.env_name == 'MPRoom':
                        map_id = [2, 3, 6, 7, 9]
                        spline = False
                        app_id = np.random.choice(map_id)
                    else:
                        map_id = [1, 2, 3, 4]
                        spline = True
                        app_id = np.random.choice(map_id)
                    self.unrealcv.set_appearance(obj, app_id, spline)

        # random light and texture of the agents
        if player_texture:
            if self.env_name == 'MPRoom':  # random target texture
                for obj in self.player_list:
                    if self.agents[obj]['agent_type'] == 'player':
                        self.unrealcv.random_player_texture(obj, self.textures_list, 3)
        if light:
            self.unrealcv.random_lit(self.env_configs["lights"])

        # random the texture of the background
        if background_texture:
            self.unrealcv.random_texture(self.env_configs["backgrounds"], self.textures_list, 3)

        # random place the obstacle
        if layout:
            self.unrealcv.clean_obstacles()
            self.unrealcv.random_obstacles(self.objects_list, self.textures_list,
                                           15, self.reset_area, self.start_area, layout_texture)

    def get_pose_states(self, obj_pos):
        # get the relative pose of each agent and the absolute location and orientation of the agent
        pose_obs = []
        player_num = len(obj_pos)
        np.zeros((player_num, player_num, 2))
        relative_pose = np.zeros((player_num, player_num, 2))
        for j in range(player_num):
            vectors = []
            for i in range(player_num):
                obs, distance, direction = self.get_relative(obj_pos[j], obj_pos[i])
                yaw = obj_pos[j][4]/180*np.pi
                # rescale the absolute location and orientation
                abs_loc = [obj_pos[i][0]/self.exp_distance, obj_pos[i][1]/self.exp_distance,
                           obj_pos[i][2]/self.exp_distance, np.cos(yaw), np.sin(yaw)]
                obs = obs + abs_loc
                vectors.append(obs)
                relative_pose[j, i] = np.array([distance, direction])
            pose_obs.append(vectors)

        return np.array(pose_obs), relative_pose

    def launch_ue_env(self):
        # launch the UE4 binary and connect to UnrealCV
        env_ip, env_port = self.unreal.start(self.docker, self.resolution,
                                             self.display, self.use_opengl,
                                             self.offscreen_rendering, self.nullrhi)
        # connect UnrealCV
        self.unrealcv = Tracking(cam_id=self.cam_id[0], port=env_port, ip=env_ip,
                                 env=self.unreal.path2env, resolution=self.resolution)

        return True

    def init_agents(self):
        for obj in self.player_list.copy(): # the agent will be fully removed in self.agents
            if self.agents[obj]['agent_type'] == 'car':
                # self.unrealcv.set_obj_scale(obj, [0.5, 0.5, 0.5])
                self.remove_agent(obj)
            elif self.agents[obj]['agent_type'] == 'drone':
                self.remove_agent(obj)
            elif self.agents[obj]['agent_type'] == 'animal':
                self.remove_agent(obj)

        for obj in self.player_list:
            self.agents[obj]['scale'] = self.unrealcv.get_obj_scale(obj)
            self.unrealcv.set_random(obj, 0)
            self.unrealcv.set_interval(self.interval, obj)

        self.unrealcv.build_color_dic(self.player_list)
        self.cam_flag = self.unrealcv.get_cam_flag(self.observation_type)

    def init_objects(self):
        if self.reset_type >= 4:
            self.unrealcv.init_objects(self.objects_list)

    def prepare_img2show(self, index, states):
        if self.observation_type == 'Rgbd':
            return states[index][:, :, :3]
        elif self.observation_type in ['Color', 'Depth', 'Gray', 'CG', 'Mask']:
            return states[index]
        else:
            return None

    def adjust_agents(self, num_agents):
        while len(self.player_list) < num_agents:
            refer_agent = self.agents[random.choice(list(self.agents.keys()))]
            name = f'{refer_agent["agent_type"]}_EP{self.count_eps}_{len(self.player_list)}'
            self.agents[name] = self.add_agent(name, random.choice(self.safe_start), refer_agent)
        while len(self.player_list) > num_agents:
            self.remove_agent(self.player_list[-1])  # remove the last one

    def sample_target(self):
        target_list = self.player_list.copy()
        target_list.pop(self.tracker_id)
        target_id = self.player_list.index(random.choice(target_list))
        return target_id

    def sample_tracker(self):
        return self.cam_list.index(random.choice([x for x in self.cam_list if x > 0]))
