import math
import os
import time
import gym
import numpy as np
from gym import spaces
from gym_unrealcv.envs.tracking import reward
from gym_unrealcv.envs.tracking.visualization import show_info
from gym_unrealcv.envs.utils import env_unreal
from gym_unrealcv.envs.navigation.interaction import Navigation
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
                reset_type = 'random',       # random
                action_type = 'discrete',  # 'discrete', 'continuous'
                observation_type = 'color', # 'color', 'depth', 'rgbd'
                reward_type = 'distance', # distance
                docker = False,
                resolution=(84, 84)
                ):

     setting = self.load_env_setting(setting_file)
     self.docker = docker
     self.reset_type = reset_type
     self.roll = 0
     self.pitch = 0

     # start unreal env
     self.unreal = env_unreal.RunUnreal(ENV_BIN=setting['env_bin'])
     env_ip, env_port = self.unreal.start(docker,resolution)

     # connect UnrealCV
     self.unrealcv = Navigation(cam_id=self.cam_id,
                                port= env_port,
                                ip=env_ip,
                                env=self.unreal.path2env,
                                resolution=resolution)

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
     self.observation_space = self.unrealcv.define_observation(self.cam_id,self.observation_type)

     # define reward type
     # distance, bbox, bbox_distance,
     self.reward_type = reward_type
     self.reward_function = reward.Reward(setting)


     self.rendering = False

     # init augment env
     if 'random' in self.reset_type or 'hide' in self.reset_type:
         self.show_list = self.objects_env
         self.hiden_list = random.sample(self.objects_env, min(15,len(self.objects_env)))
         for x in self.hiden_list:
            self.show_list.remove(x)
            self.unrealcv.hide_obj(x)

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
        action = np.squeeze(action)
        if self.action_type == 'discrete':
            # linear
            (velocity, angle) = self.discrete_actions[action]
        else:
            (velocity, angle) = action

        info['Collision'] = self.unrealcv.move_2d(self.cam_id, angle, velocity)

        self.count_steps += 1

        info['Pose'] = self.unrealcv.get_pose(self.cam_id)
        self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])
        info['Direction'] = self.get_direction(self.target_pos, info['Pose'][:3]) - info['Pose'][-2]
        if info['Direction'] < -180:
            info['Direction'] += 360
        elif info['Direction'] > 180:
            info['Direction'] -= 360

        info['Distance'] = self.get_distance(self.target_pos,info['Pose'][:3])

        # update observation
        state = self.unrealcv.get_observation(self.cam_id, self.observation_type)
        info['Color'] = self.unrealcv.img_color
        info['Depth'] = self.unrealcv.img_depth

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
           print ('Reach Max Steps')

        # save the trajectory
        self.trajectory.append(info['Pose'])
        info['Trajectory'] = self.trajectory

        if self.rendering:
            show_info(info,self.action_type)

        self.C_reward += info['Reward']
        return state, info['Reward'], info['Done'], info
   def _reset(self, ):
       self.C_reward = 0

       self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])
       # random hide and show objects
       if 'random' in self.reset_type:
           num_update = 3
           objs_to_hide = random.sample(self.show_list,num_update)
           for x in objs_to_hide:
               self.show_list.remove(x)
               self.hiden_list.append(x)
               self.unrealcv.hide_obj(x)
           #self.unrealcv.hide_objects(to_hiden)
           objs_to_show = random.sample(self.hiden_list[:-num_update],num_update)
           for x in objs_to_show:
               self.hiden_list.remove(x)
               self.show_list.append(x)
               self.unrealcv.show_obj(x)
           #self.unrealcv.show_objects(to_show)
           time.sleep(0.5 * random.random())

       time.sleep(0.5)

       cam_pos = self.target_pos
       self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])
       yaw = self.get_direction(self.target_pos, cam_pos)
       #self.reward_function.dis_exp = 1.5 * self.get_distance(self.target_pos,cam_pos[:3])
       #print self.reward_function.dis_exp
       # set pose
       self.unrealcv.set_location(self.cam_id, cam_pos)
       self.unrealcv.set_rotation(self.cam_id, [self.roll, yaw, self.pitch])
       current_pose = self.unrealcv.get_pose(self.cam_id,'soft')

       # get state
       time.sleep(0.1)
       state = self.unrealcv.get_observation(self.cam_id, self.observation_type)

       self.trajectory = []
       self.trajectory.append(current_pose)
       self.count_steps = 0

       return state

   def _close(self):
       if self.docker:
           self.unreal.docker.close()

   def _seed(self, seed=None):
       print('fake seed')

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
           print ('unknown type')

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
