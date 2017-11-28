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
import random

'''
It is an env for active object tracking.

State : raw color image and depth
Action:  (linear velocity ,angle velocity) 
Done : the relative distance or angle to target is larger than the threshold.
Task: Learn to follow the target object(moving person) in the scene
'''



class UnrealCvTracking_base(gym.Env):
   def __init__(self,
                setting_file = 'search_quadcopter.json',
                reset_type = 'waypoint',       # testpoint, waypoint,
                test = True,               # if True will use the test_xy as start point
                action_type = 'discrete',  # 'discrete', 'continuous'
                observation_type = 'rgbd', # 'color', 'depth', 'rgbd'
                reward_type = 'distance', # distance
                docker = False,
                ):

     print setting_file
     setting = self.load_env_setting(setting_file)
     self.test = test
     self.docker = docker
     self.reset_type = reset_type
     self.roll = 0
     self.pitch = 0

     # start unreal env
     self.unreal = env_unreal.RunUnreal(ENV_BIN=setting['env_bin'])
     env_ip, env_port = self.unreal.start(docker)

     # connect UnrealCV
     self.unrealcv = Tracking(cam_id=self.cam_id,
                              port= env_port,
                              ip=env_ip,
                              env=self.unreal.path2env)

    # define action
     self.action_type = action_type
     assert self.action_type == 'discrete' or self.action_type == 'continuous'
     if self.action_type == 'discrete':
         self.action_space = spaces.Discrete(len(self.discrete_actions))
     elif self.action_type == 'continuous':
         self.action_space = spaces.Box(low = np.array(self.continous_actions['low']),high = np.array(self.continous_actions['high']))

     self.count_steps = 0


    # define observation space,
    # color, depth, rgbd,...
     self.observation_type = observation_type
     assert self.observation_type == 'color' or self.observation_type == 'depth' or self.observation_type == 'rgbd'
     if self.observation_type == 'color':
         state = self.unrealcv.read_image(self.cam_id,'lit')
         self.observation_space = spaces.Box(low=0, high=255, shape=state.shape)
     elif self.observation_type == 'depth':
         state = self.unrealcv.read_depth(self.cam_id)
         self.observation_space = spaces.Box(low=0, high=10, shape=state.shape)
     elif self.observation_type == 'rgbd':
         state = self.unrealcv.get_rgbd(self.cam_id)
         s_high = state
         s_high[:,:,-1] = 10.0
         s_high[:,:,:-1] = 255
         s_low = np.zeros(state.shape)
         self.observation_space = spaces.Box(low=s_low, high=s_high)

     # define reward type
     # distance, bbox, bbox_distance,
     self.reward_type = reward_type
     self.reward_function = reward.Reward(setting)


     # set start position
     self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])
     time.sleep(0.5 + 0.3*random.random())
     cam_pos = self.target_pos
     self.unrealcv.set_location(self.cam_id,cam_pos)
     self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])
     yaw = self.get_direction(self.target_pos, cam_pos)
     self.unrealcv.set_rotation(self.cam_id,[self.roll,yaw,self.pitch])

     self.trajectory = []

     self.rendering = False

     if 'random' in self.reset_type:
         self.show_list = self.objects_env
         self.hiden_list = random.sample(self.objects_env, 15)

         for x in self.hiden_list:
            self.show_list.remove(x)


   def _render(self, mode='human', close=False):
       self.rendering = True

   def _step(self, action ):
        info = dict(
            Collision=False,
            Done = False,
            Trigger=0.0,
            Maxstep=False,
            Reward=0.0,
            Action = action,
            Pose = [],
            Trajectory = self.trajectory,
            Steps = self.count_steps,
            Direction = None,
            Distance = None,
            Color = None,
            Depth = None,
        )

        if self.action_type == 'discrete':
            # linear
            (velocity, angle) = self.discrete_actions[action]
        else:
            (velocity, angle) = action

        info['Collision'] = self.unrealcv.move_2d(self.cam_id, angle, velocity)

        self.count_steps += 1

        info['Pose'] = self.unrealcv.get_pose(self.cam_id)
        self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])
        info['Direction'] = self.get_direction(self.target_pos, info['Pose'][:3]) - info['Pose'][-1]
        if info['Direction'] < -180:
            info['Direction'] += 360
        elif info['Direction'] > 180:
            info['Direction'] -= 360

        info['Distance'] = self.get_distance(self.target_pos,info['Pose'][:3])

        # update observation
        if self.observation_type == 'color':
            state = info['Color'] = self.unrealcv.read_image(self.cam_id, 'lit')
        elif self.observation_type == 'depth':
            state = info['Depth'] = self.unrealcv.read_depth(self.cam_id)
        elif self.observation_type == 'rgbd':
            info['Color'] = self.unrealcv.read_image(self.cam_id, 'lit')
            info['Depth'] = self.unrealcv.read_depth(self.cam_id)
            state = np.append(info['Color'], info['Depth'], axis=2)


        if info['Distance'] > self.max_distance or abs(info['Direction'])> self.max_direction:
        #if self.C_reward<-450: # for evaluation
            info['Done'] = True
            info['Reward'] = -1
        elif 'distance' in self.reward_type:
            info['Reward'] = self.reward_function.reward_distance(info['Distance'],info['Direction'])


        # limit the max steps of every episode
        if self.count_steps > self.max_steps:
           info['Done'] = True
           info['Maxstep'] = True
           print 'Reach Max Steps'

        # save the trajectory
        self.trajectory.append(info['Pose'])
        info['Trajectory'] = self.trajectory

        if self.rendering:
            show_info(info,self.action_type)

        self.C_reward += info['Reward']
        return state, info['Reward'], info['Done'], info
   def _reset(self, ):
       self.C_reward = 0

       # random env
       if 'random' in self.reset_type:
           num_update = 3
           to_hiden = random.sample(self.show_list,num_update)
           for x in to_hiden:
               self.show_list.remove(x)
               self.hiden_list.append(x)
           self.unrealcv.hide_objects(to_hiden)

           to_show = random.sample(self.hiden_list[:-num_update],num_update)
           for x in to_show:
               self.hiden_list.remove(x)
               self.show_list.append(x)
           self.unrealcv.show_objects(to_show)




       #self.unrealcv.keyboard('R') #reset target object
       self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])
       time.sleep(0.5 + 0.3*random.random())
       cam_pos = self.target_pos
       self.unrealcv.set_location(self.cam_id, [cam_pos[0], cam_pos[1], cam_pos[2]])
       self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])
       yaw = self.get_direction(self.target_pos, cam_pos)
       self.unrealcv.set_rotation(self.cam_id, [self.roll, yaw, self.pitch])
       current_pose = self.unrealcv.get_pose(self.cam_id)


       if self.observation_type == 'color':
           state = self.unrealcv.read_image(self.cam_id, 'lit')
       elif self.observation_type == 'depth':
           state = self.unrealcv.read_depth(self.cam_id)
       elif self.observation_type == 'rgbd':
           state = self.unrealcv.get_rgbd(self.cam_id)


       self.trajectory = []
       self.trajectory.append(current_pose)
       self.count_steps = 0

       return state

   def _close(self):
       if self.docker:
           self.unreal.docker.close()


   def _get_action_size(self):
       return len(self.action)


   def get_distance(self,target,current):

       error = abs(np.array(target)[:2] - np.array(current)[:2])# only x and y
       distance = math.sqrt(sum(error * error))
       return distance


   def get_direction(self,target_pos,camera_pos):
       relative_pos = np.array(target_pos) - np.array(camera_pos)
       if relative_pos[0] > 0:
           direction = 180 * np.arctan(relative_pos[1] / relative_pos[0]) / np.pi
       else:
           direction = 180 + 180 * np.arctan(relative_pos[1] / relative_pos[0]) / np.pi

       return direction


   def load_env_setting(self,filename):
       f = open(self.get_settingpath(filename))
       type = os.path.splitext(filename)[1]
       if type == '.json':
           import json
           setting = json.load(f)
       elif type == '.yaml':
           import yaml
           setting = yaml.load(f)
       else:
           print 'unknown type'

       #print setting
       self.cam_id = setting['cam_id']
       self.target_list = setting['targets']
       self.max_steps = setting['max_steps']
       self.discrete_actions = setting['discrete_actions']
       self.continous_actions = setting['continous_actions']
       self.max_distance = setting['max_distance']
       self.max_direction = setting['max_direction']
       self.objects_env = setting['objects_list']

       return setting


   def get_settingpath(self, filename):
       import gym_unrealcv
       gympath = os.path.dirname(gym_unrealcv.__file__)
       return os.path.join(gympath, 'envs/setting', filename)
