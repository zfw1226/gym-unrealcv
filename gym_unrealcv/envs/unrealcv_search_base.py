import math
import os
import time

import gym
import numpy as np
from gym import spaces
from gym_unrealcv.envs.navigation import reward, reset_point
from gym_unrealcv.envs.navigation.visualization import show_info
from gym_unrealcv.envs.utils import env_unreal
from gym_unrealcv.envs.navigation.interaction import Navigation
import random
'''
It is a general env for searching target object.

State : raw color image and depth (640x480) 
Action:  (linear velocity ,angle velocity , trigger) 
Done : Collision or get target place or False trigger three times.
Task: Learn to avoid obstacle and search for a target object in a room, 
      you can select the target name according to the Recommend object list as below

Recommend object list in RealisticRendering
 'SM_CoffeeTable_14', 'Couch_13','SM_Couch_1seat_5','Statue_48','SM_TV_5', 'SM_DeskLamp_5'
 'SM_Plant_7', 'SM_Plant_8', 'SM_Door_37', 'SM_Door_39', 'SM_Door_41'

Recommend object list in Arch1
'BP_door_001_C_0','BP_door_002_C_0'
'''

class UnrealCvSearch_base(gym.Env):
   def __init__(self,
                setting_file = 'search_rr_plant78.json',
                reset_type = 'waypoint',       # testpoint, waypoint,
                augment_env = None,   #texture, target, light
                test = True,               # if True will use the test_xy as start point
                action_type = 'discrete',  # 'discrete', 'continuous'
                observation_type = 'rgbd', # 'color', 'depth', 'rgbd'
                reward_type = 'bbox', # distance, bbox, bbox_distance,
                docker = False,
                resolution = (320,240)
                ):

     setting = self.load_env_setting(setting_file)
     self.test = test
     self.docker = docker
     self.reset_type = reset_type
     self.augment_env = augment_env

     # start unreal env
     self.unreal = env_unreal.RunUnreal(ENV_BIN=setting['env_bin'])
     env_ip,env_port = self.unreal.start(docker,resolution)

     # connect UnrealCV
     self.unrealcv = Navigation(cam_id=self.cam_id,
                              port= env_port,
                              ip=env_ip,
                              targets=self.target_list,
                              env=self.unreal.path2env,
                              resolution=resolution)
     self.unrealcv.pitch = self.pitch

     # define action
     self.action_type = action_type
     assert self.action_type == 'discrete' or self.action_type == 'continuous'
     if self.action_type == 'discrete':
         self.action_space = spaces.Discrete(len(self.discrete_actions))
     elif self.action_type == 'continuous':
         self.action_space = spaces.Box(low = np.array(self.continous_actions['low']),high = np.array(self.continous_actions['high']))

     # define observation space,
     # color, depth, rgbd,...
     self.observation_type = observation_type
     assert self.observation_type == 'color' or self.observation_type == 'depth' or self.observation_type == 'rgbd'
     self.observation_shape = self.unrealcv.define_observation(self.cam_id,self.observation_type)

     # define reward type
     # distance, bbox, bbox_distance,
     self.reward_type = reward_type
     self.reward_function = reward.Reward(setting)

     # set start position
     self.trigger_count  = 0
     current_pose = self.unrealcv.get_pose(self.cam_id)
     current_pose[2] = self.height
     self.unrealcv.set_location(self.cam_id,current_pose[:3])

     self.count_steps = 0
     self.targets_pos = self.unrealcv.build_pose_dic(self.target_list)

     # for reset point generation and selection
     self.reset_module = reset_point.ResetPoint(setting, reset_type, test, current_pose)

     self.rendering = False

   def _step(self, action ):
        info = dict(
            Collision=False,
            Done = False,
            Trigger=0.0,
            Maxstep=False,
            Reward=0.0,
            Action = action,
            Bbox =[],
            Pose = [],
            Trajectory = self.trajectory,
            Steps = self.count_steps,
            Target = [],
            Direction = None,
            Waypoints = self.reset_module.waypoints,
            Color = None,
            Depth = None,
        )


        if self.action_type == 'discrete':
            (velocity, angle, info['Trigger']) = self.discrete_actions[action]
        else:
            (velocity, angle, info['Trigger']) = action
        self.count_steps += 1
        info['Done'] = False

        # take action
        info['Collision'] = self.unrealcv.move_2d(self.cam_id, angle, velocity)
        info['Pose'] = self.unrealcv.get_pose(self.cam_id, 'soft')

        # the robot think that it found the target object,the episode is done
        # and get a reward by bounding box size
        # only three times false trigger allowed in every episode
        if info['Trigger'] > self.trigger_th :
            self.trigger_count += 1
            # get reward
            if 'bbox' in self.reward_type:
                object_mask = self.unrealcv.read_image(self.cam_id, 'object_mask')
                boxes = self.unrealcv.get_bboxes(object_mask, self.target_list)
                info['Reward'], info['Bbox'] = self.reward_function.reward_bbox(boxes)
            else:
                info['Reward'] = 0

            if info['Reward'] > 0 or self.trigger_count > 3:
                info['Done'] = True
                if info['Reward'] > 0 and self.test == False:
                    self.reset_module.success_waypoint(self.count_steps)
                print ('Trigger Terminal!')

        else :
            # get reward
            distance, self.target_id = self.select_target_by_distance(info['Pose'][:3],self.targets_pos)
            info['Target'] = self.targets_pos[self.target_id]
            info['Direction'] = self.get_direction(info['Pose'], self.targets_pos[self.target_id])

            # calculate reward according to the distance to target object
            if 'distance' in self.reward_type:
                info['Reward'] = self.reward_function.reward_distance(distance)
            else:
                info['Reward'] = 0

            # if collision detected, the episode is done and reward is -1
            if info['Collision']:
                info['Reward'] = -1
                info['Done'] = True
                self.reset_module.update_dis2collision(info['Pose'])
                print ('Collision!!')

        # update observation
        state = self.unrealcv.get_observation(self.cam_id, self.observation_type)
        info['Color'] = self.unrealcv.img_color
        info['Depth'] = self.unrealcv.img_depth

        # limit the max steps of every episode
        if self.count_steps > self.max_steps:
           info['Done'] = True
           info['Maxstep'] = True
           print ('Reach Max Steps')
        # save the trajectory
        self.trajectory.append(info['Pose'][:6])
        info['Trajectory'] = self.trajectory
        if info['Done'] and len(self.trajectory) > 5 and not self.test:
            self.reset_module.update_waypoint(info['Trajectory'])

        if self.rendering:
            show_info(info)

        return state, info['Reward'], info['Done'], info
   def _reset(self, ):
       # augment environment
       self.random_scene(self.augment_env)

       # double check the resetpoint
       collision = True
       while collision:
           current_pose = self.reset_module.select_resetpoint()
           self.unrealcv.set_pose(self.cam_id, current_pose)
           collision = self.unrealcv.move_2d(self.cam_id, 0, 100)
       self.unrealcv.set_pose(self.cam_id,current_pose)

       state = self.unrealcv.get_observation(self.cam_id,self.observation_type)

       self.trajectory = []
       self.trajectory.append(current_pose)
       self.trigger_count = 0
       self.count_steps = 0
       self.reward_function.dis2target_last, self.targetID_last = self.select_target_by_distance(current_pose,
                                                                                                 self.targets_pos)
       return state

   def _close(self):
       if self.docker:
           self.unreal.docker.close()


   def _get_action_size(self):
       return len(self.action)

   def random_scene(self,augment_env):
       if isinstance(augment_env, str):
           # change material
           if 'texture' in augment_env:
               num = ['One', 'Two', 'Three', 'Four', 'Five','Six']
               for i in num:
                   self.unrealcv.keyboard(i)

           if 'target' in augment_env:
               for i in self.target_list:
                   z1 = 60
                   while z1 > 50:
                       x = random.uniform(self.reset_area[0], self.reset_area[1])
                       y = random.uniform(self.reset_area[2], self.reset_area[3])
                       self.unrealcv.set_object_location(i, [x, y, 50])
                       time.sleep(0.5)
                       [x1, y1, z1] = self.unrealcv.get_obj_location(i)


   def get_distance(self,target,current):

       error = abs(np.array(target)[:2] - np.array(current)[:2])# only x and y
       distance = math.sqrt(sum(error * error))
       return distance


   def select_target_by_distance(self,current_pos, targets_pos):
       # find the nearest target, return distance and targetid
       target_id = self.targets_pos.keys()[0]
       distance_min = self.get_distance(targets_pos[target_id], current_pos)
       for key, target_pos in targets_pos.items():
           distance = self.get_distance(target_pos, current_pos)
           if distance < distance_min:
               target_id = key
               distance_min = distance

       return distance_min,target_id

   def get_direction(self,current_pose,target_pose):
       y_delt = target_pose[1] - current_pose[1]
       x_delt = target_pose[0] - current_pose[0]
       if x_delt == 0:
           x_delt = 0.00001

       angle_now = np.arctan(y_delt / x_delt) / 3.1415926 * 180 - current_pose[-1]

       if x_delt < 0:
           angle_now += 180
       if angle_now < 0:
           angle_now += 360
       if angle_now > 360:
           angle_now -= 360

       return angle_now


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

       self.cam_id = setting['cam_id']
       self.target_list = setting['targets']
       self.max_steps = setting['maxsteps']
       self.trigger_th = setting['trigger_th']
       self.height = setting['height']
       self.pitch = setting['pitch']

       self.discrete_actions = setting['discrete_actions']
       self.continous_actions = setting['continous_actions']

       return setting


   def get_settingpath(self, filename):
       import gym_unrealcv
       gympath = os.path.dirname(gym_unrealcv.__file__)
       return os.path.join(gympath, 'envs/setting', filename)


