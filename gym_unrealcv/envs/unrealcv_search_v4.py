import gym
from unrealcv_cmd import  UnrealCv
import cv2
import numpy as np
import time
import random
import math
import run_docker
from gym import spaces
import yaml
import sys
import os
'''
 
'''

'''
It is a general env for searching target object.

State : raw color image (640x480)
Action:  (linear velocity ,angle velocity , trigger) /continuous space
Done : Collision or get target place
Task: Learn to avoid obstacle and search for a target object in a room, 
      you can select the target name according to the Recommend object list as below

Recommend object list in RealisticRendering
 'SM_CoffeeTable_14', 'Couch_13','SM_Couch_1seat_5','Statue_48','SM_TV_5', 'SM_DeskLamp_5'
 'SM_Plant_7', 'SM_Plant_8', 'SM_Door_37', 'SM_Door_39', 'SM_Door_41'
 
Recommend object list in Arch1
'BP_door_001_C_0','BP_door_002_C_0'
'''

class UnrealCvSearch_v4(gym.Env):
   def __init__(self,
                setting_file = 'search_rr_plant78.yaml'
                ):

     setting = self.load_env_setting(setting_file)

     if setting['docker']:
         self.docker = run_docker.RunDocker()
         env_ip, env_dir = self.docker.start(ENV_NAME = setting['env_name'])
         self.unrealcv = UnrealCv(self.cam_id, ip=env_ip, targets=self.target_list, env=env_dir)
     else:
         self.docker = False
         #you need run the environment previously
         env_ip = '127.0.0.1'
         self.unrealcv = UnrealCv(self.cam_id, ip=env_ip, targets=self.target_list)

     print env_ip

     self.action = (0,0,0) #(velocity,angle,trigger)
     self.action_space = spaces.Box(low = 0,high = 100, shape = self.action)

     self.count_steps = 0
     self.targets_pos = self.unrealcv.get_objects_pos(self.target_list)
     self.trajectory = []

     state = self.unrealcv.read_image(self.cam_id, 'lit')
     self.observation_space = spaces.Box(low=0, high=255, shape=state.shape)


     self.trigger_count  = 0


   def _step(self, action = (0,0,0), show = False):
        info = dict(
            Collision=False,
            Done = False,
            Trigger=0.0,
            Maxstep=False,
            Reward=0.0,
            Action = action,
            Bbox =[],
            Pose = [],
            Trajectory = [],
            Steps = self.count_steps,
            Target = self.targets_pos[0]
        )

        (velocity, angle, info['Trigger']) = action
        self.count_steps += 1
        info['Done'] = False

        # the robot think that it found the target object,the episode is done
        # and get a reward by bounding box size
        # only three times false trigger allowed in every episode
        if info['Trigger'] > self.trigger_th :
            #self.unrealcv.set_rotation(self.cam_id, self.unrealcv.cam['rotation'][0], self.unrealcv.cam['rotation'][1],20)
            state = self.unrealcv.read_image(self.cam_id, 'lit', show=False)
            self.trigger_count += 1
            info['Reward'],info['Bbox'] = self.reward_bbox()
            if info['Reward'] > 0 or self.trigger_count > 3:
                info['Done'] = True
                print 'Trigger Terminal!'

        # if collision occurs, the episode is done and reward is -1
        else :
            info['Collision'] = self.unrealcv.move(self.cam_id, angle, velocity)
            state = self.unrealcv.read_image(self.cam_id, 'lit', show=False)
            info['Reward'] = 0
            if info['Collision']:
                info['Reward'] = -1
                info['Done'] = True
                print ('Collision!!')

        # limit the max steps of every episode
        if self.count_steps > self.max_steps:
           info['Done'] = True
           info['Maxstep'] = True
           print 'Reach Max Steps'

        # save the trajectory
        info['Pose'] = self.unrealcv.get_pose()
        self.trajectory.append(info['Pose'])
        info['Trajectory'] = self.trajectory

        if show:
            self.unrealcv.show_img(state, 'state')

        return state, info['Reward'], info['Done'], info
   def _reset(self, ):
       # set a random start point according to the origin list
       self.count_steps = 0
       start_id = random.randint(0, len(self.start_xy)-1)
       x,y = self.start_xy[start_id]
       self.unrealcv.set_position(self.cam_id, x, y, self.height)
       self.unrealcv.set_rotation(self.cam_id, 0, random.randint(0,360), 0)
       state = self.unrealcv.read_image(self.cam_id , 'lit')

       current_pos = self.unrealcv.get_pos()
       self.trajectory = []
       self.trajectory.append(current_pos)
       self.trigger_count = 0

       return state

   def _close(self):
       if self.docker:
           self.docker.close()

   def _get_action_size(self):
       return len(self.action)


   def reward_bbox(self):

       object_mask = self.unrealcv.read_image(self.cam_id, 'object_mask')
       #mask, box = self.unrealcv.get_bbox(object_mask, self.target_list[0])
       boxes = self.unrealcv.get_bboxes(object_mask,self.target_list)
       reward = 0
       for box in boxes:
           reward += self.calculate_bbox_reward(box)

       if reward > self.reward_th:
            reward = reward * 10
            print ('Get ideal Target!!!')
       elif reward == 0:
           reward = -1
           print ('Get Nothing')
       else:
           reward = 0
           print ('Get small Target!!!')

       return reward,boxes


   def calculate_bbox_reward(self,box):
       (xmin,ymin),(xmax,ymax) = box
       boxsize = (ymax - ymin) * (xmax - xmin)
       x_c = (xmax + xmin) / 2.0
       x_bias = x_c - 0.5
       discount = max(0, 1 - x_bias ** 2)
       reward = discount * boxsize
       return reward

   def load_env_setting(self,filename):
       f = open(self.get_abspath(filename))
       setting = yaml.load(f)
       print setting
       self.cam_id = setting['cam_id']
       self.target_list = setting['targets']
       self.max_steps = setting['maxsteps']
       self.reward_th = setting['reward_th']
       self.trigger_th = setting['trigger_th']
       self.height = setting['height']
       self.start_xy = setting['start_xy']
       return setting

   def get_abspath(self, filename):
       paths = sys.path
       for p in paths:
           if p[-20:].find('gym-unrealcv') > 0:
               gympath = p
       return os.path.join(gympath, 'gym_unrealcv/envs/setting', filename)