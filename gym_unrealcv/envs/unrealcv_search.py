import gym
from unrealcv_cmd import  UnrealCv
import cv2
import numpy as np
import time
import random
import math
import run_docker
from gym import spaces

'''
State : raw color image (640x480)
Action:  (linear velocity ,angle velocity , trigger) /continuous space
Done : Collision or get target place
Task: Learn to avoid obstacle and search for a target object in a room, 
      you can select the target name according to the Recommend object list as below

Recommend object list
 'SM_CoffeeTable_14', 'Couch_13','SM_Couch_1seat_5','Statue_48','SM_TV_5', 'SM_DeskLamp_5'
 'SM_Plant_7', 'SM_Plant_8', 'SM_Door_37', 'SM_Door_39', 'SM_Door_41'
'''

class UnrealCvSearch(gym.Env):
   def __init__(self,
                TARGETS = ['SM_Plant_8','SM_Plant_7'],
                DOCKER = True,
                ENV_NAME='RealisticRendering',
                cam_id = 0,
                MAX_STEPS = 100):
     self.cam_id = cam_id
     self.target_list = TARGETS

     if DOCKER:
         self.docker = run_docker.RunDocker()
         env_ip, env_dir = self.docker.start(ENV_NAME = ENV_NAME)
         self.unrealcv = UnrealCv(self.cam_id, ip=env_ip, targets=self.target_list, env=env_dir)
     else:
         self.docker = False
         #you need run the environment previously
         env_ip = '127.0.0.1'
         self.unrealcv = UnrealCv(self.cam_id, ip=env_ip, targets=self.target_list)

     print env_ip


     self.reward_th = 0.2 # plant 0.2
     self.trigger_th = 0.9
     height = 40
     self.origin = [
         (-106.195,  437.424,   height),
         (  27.897, -162.437,   height),
         ( -66.601,   -4.503,   height),
         (  10.832,  135.126,   height),
         (  67.903,   26.995,   height),
         ( -23.558, -187.354,   height),
         ( -80.312,  254.579,   height),
     ]

     self.count_steps = 0
     self.max_steps = MAX_STEPS
     self.action = (0,0,0) #(velocity,angle,trigger)

     self.targets_pos = self.unrealcv.get_objects_pos(self.target_list)
     self.trajectory = []

     state = self.unrealcv.read_image(self.cam_id, 'lit')

     #self.action_space = spaces.Discrete(len(self.ACTION_LIST))
     self.action_space = spaces.Box(low = 0,high = 100, shape = self.action)
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
       start_point = random.randint(0, len(self.origin)-1)
       x,y,z = self.origin[start_point]
       self.unrealcv.set_position(self.cam_id, x, y, z)
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

