import math
import os
import random
import time

import gym
import numpy as np
from gym import spaces
from gym_unrealcv.envs.robotarm.visualization import show_info
from gym_unrealcv.envs.utils import env_unreal
from gym_unrealcv.envs.utils.unrealcv_cmd import UnrealCv


class UnrealCvRobotArm_base(gym.Env):
   def __init__(self,
                setting_file = 'robotarm_v3.json',
                reset_type = 'keyboard',    # testpoint, waypoint,
                test = True,                # if True will use the test_xy as start point
                action_type = 'discrete',   # 'discrete', 'continuous'
                observation_type = 'color', # 'color', 'depth', 'rgbd'
                reward_type = 'move', # distance, move, move_distance
                docker = False,
                ):

     setting = self.load_env_setting(setting_file)
     self.test = test
     self.docker = docker
     self.reset_type = reset_type

     # start unreal env
     self.unreal = env_unreal.RunUnreal(ENV_BIN=setting['env_bin'])
     env_ip,env_port = self.unreal.start(docker)


     # connect UnrealCV
     self.unrealcv = UnrealCv(cam_id=self.cam_id,
                              port= env_port,
                              ip=env_ip,
                              targets=self.target_list,
                              env=self.unreal.path2env)

    # define action type
     self.action_type = action_type
     assert self.action_type == 'discrete' or self.action_type == 'continuous'
     if self.action_type == 'discrete':
         self.action_space = spaces.Discrete(len(self.discrete_actions))
     elif self.action_type == 'continuous':
         self.action_low = np.array(self.continous_actions['low'])
         self.action_high = np.array(self.continous_actions['high'])
         self.action_space = spaces.Box(low = self.action_low,high = self.action_high)

     self.pose_low = np.array(self.pose_range['low'])
     self.pose_high = np.array(self.pose_range['high'])

     self.count_steps = 0


    # define observation space,
    # color, depth, rgbd...
     self.observation_type = observation_type
     assert self.observation_type == 'color' or self.observation_type == 'depth' or self.observation_type == 'rgbd' or self.observation_type == 'measured'
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
     elif self.observation_type == 'measured':
         s_high = [85,  80,  90,  95,  120, 200, 300, 360, 250,  400, 360] # arm_pose, grip_position, target_position
         s_low = [ 0, -90, -60, -55, -120, -400, -150,  0, -350, -150,  40]
         self.observation_space = spaces.Box(low=np.array(s_low), high=np.array(s_high))

     # define reward type
     # distance, bbox, bbox_distance,
     self.reward_type = reward_type
     self.rendering = False



   def _step(self, action):
        info = dict(
            Collision=False,
            Done = False,
            Maxstep=False,
            Reward=0.0,
            Action = action,
            Bbox =self.bboxes,
            ArmPose = [],
            GripPosition = [],
            Steps= self.count_steps+1,
            Target = self.target_list,
            TargetPose = self.target_pose,
            Color = None,
            Depth = None,
        )

        self.unrealcv.message = []

        # take a action
        if self.action_type == 'discrete':
            #duration = max(0.05, 0.2 + 0.1 * np.random.randn())
            duration = 0.05
            self.unrealcv.keyboard(self.discrete_actions[action], duration=duration)

        elif self.action_type == 'continuous':
            for i in range(len(action)):
                if abs(action[i]) < 0.05:
                    continue
                if action[i] > 0:
                    self.unrealcv.keyboard(self.discrete_actions[i*2],duration=max(action[i],0.5))
                else:
                    self.unrealcv.keyboard(self.discrete_actions[i*2 + 1],duration=max(abs(action[i]),0.5))
            duration = abs(np.array(abs(action))).max()

        time.sleep(duration)

        self.count_steps += 1
        info['Done'] = False


        info['TargetPose'] = self.unrealcv.get_object_pos(self.target_list[0])
        info['GripPosition'] = self.unrealcv.get_grip_position().tolist()

        # Get reward
        msg = self.unrealcv.read_message()
        # 'hit ground' 'ReachmaxM2' 'ReachminM2'
        if len(msg) > 0:
            print (msg)
            info['Collision'] = True
            self.count_collision += 1
            info['Done'] = False
            info['Reward'] = -1
            if self.count_collision > 3:
                info['Done'] = True
            self.target_pose = info['TargetPose']

        else:
            info['Reward'] = 0
            #target_pose_current = np.array(self.unrealcv.get_object_pos(self.target_list[0]))
            if self.get_distance(self.target_pose,info['TargetPose']) > 0.1:
                if self.count_steps > 1:
                    info['Done'] = True
                    if 'move' in self.reward_type:
                        info['Reward'] = 10
                        print ('move ball')
                self.target_pose = info['TargetPose']

            if 'distance' in self.reward_type:
                self.grip_position = np.array(info['GripPosition'])
                distance = self.get_distance(self.target_pose,self.grip_position)
                distance_delt = self.distance_last - distance
                self.distance_last = distance
                info['Reward'] = info['Reward'] + distance_delt / 100.0


        # Get observation
        if self.observation_type == 'color':
            state = info['Color'] = self.unrealcv.read_image(self.cam_id, 'lit')
            info['Depth'] = self.unrealcv.read_depth(self.cam_id)
        elif self.observation_type == 'depth':
            state = info['Depth'] = self.unrealcv.read_depth(self.cam_id)
        elif self.observation_type == 'rgbd':
            info['Color'] = self.unrealcv.read_image(self.cam_id, 'lit')
            info['Depth'] = self.unrealcv.read_depth(self.cam_id)
            state = np.append(info['Color'], info['Depth'], axis=2)
        elif self.observation_type == 'measured':
            self.arm_pose = np.array(self.unrealcv.get_arm_pose())
            state = np.append(self.arm_pose, [np.array(self.grip_position), self.target_pose])

        # bbox
        object_mask = self.unrealcv.read_image(cam_id=self.cam_id, viewmode='object_mask')
        self.bboxes = self.unrealcv.get_bboxes(object_mask=object_mask, objects=self.target_list)
        info['Bbox'] = self.bboxes

        info['ArmPose'] = self.unrealcv.get_arm_pose().tolist()

        self.arm_pose = np.array(info['ArmPose'])
        #info['ArmPose'] = (self.arm_pose - self.pose_low)/(self.pose_high-self.pose_low)
        #print info['Pose']

        # limit the max steps of every episode
        if self.count_steps > self.max_steps:
           info['Done'] = True
           info['Maxstep'] = True
           #print 'Reach Max Steps'

        if self.rendering:
            show_info(info)


        return state, info['Reward'], info['Done'], info
   def _reset(self, ):

       # set start position
       self.unrealcv.set_position(cam_id=self.cam_id, x=self.camera_pose[0][0], y=self.camera_pose[0][1],z=self.camera_pose[0][2])
       self.unrealcv.set_rotation(cam_id=self.cam_arm_id, roll=self.camera_pose[0][3], yaw=self.camera_pose[0][4],pitch=self.camera_pose[0][5])

       # for reset point generation and selection
       if self.reset_type == 'keyboard':
           self.reset_env_keyboard()


       if self.observation_type == 'color':
           state = self.unrealcv.read_image(self.cam_id, 'lit')
       elif self.observation_type == 'depth':
           state = self.unrealcv.read_depth(self.cam_id)
       elif self.observation_type == 'rgbd':
           state = self.unrealcv.get_rgbd(self.cam_id)
       elif self.observation_type == 'measured':
           self.unrealcv.message = []
           self.arm_pose = np.array(self.unrealcv.get_arm_pose())
           self.target_pose = np.array(self.unrealcv.get_object_pos(self.target_list[0]))
           self.grip_position = np.array(self.unrealcv.get_grip_position())
           state = np.append(self.arm_pose, [self.grip_position, self.target_pose])

       self.count_steps = 0
       self.count_collision = 0
       self.count_ground = 0

       self.bboxes = None
       if self.observation_type != 'measured':
           self.unrealcv.message = []
           self.target_pose = np.array(self.unrealcv.get_object_pos(self.target_list[0]))
           self.grip_position = np.array(self.unrealcv.get_grip_position())
           object_mask = self.unrealcv.read_image(cam_id=self.cam_id, viewmode='object_mask')
           self.bboxes = self.unrealcv.get_bboxes(object_mask=object_mask, objects=self.target_list)

       self.distance_last = self.get_distance(self.target_pose, self.grip_position)

       self.unrealcv.message = []

       return state

   def _close(self):
       if self.docker:
           self.unreal.docker.close()


   def _get_action_size(self):
       return len(self.action)


   def get_distance(self,target,current):

       error = abs(np.array(target)[:3] - np.array(current)[:3])# only x and y
       distance = math.sqrt(sum(error * error))
       return distance


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

       #print setting
       self.cam_id = setting['cam_view_id']
       self.cam_arm_id = setting['cam_arm_id']
       self.target_list = setting['targets']
       self.max_steps = setting['maxsteps']
       self.camera_pose = setting['camera_pose']
       self.discrete_actions = setting['discrete_actions']
       self.continous_actions = setting['continous_actions']
       self.pose_range = setting['pose_range']

       return setting


   def get_settingpath(self, filename):
       import gym_unrealcv
       gympath = os.path.dirname(gym_unrealcv.__file__)
       return os.path.join(gympath, 'envs/setting', filename)


   def reset_env_keyboard(self):
       self.unrealcv.keyboard('R')  # reset arm pose
       time.sleep(0.1)
       self.unrealcv.keyboard('LeftBracket')
     #  self.unrealcv.keyboard('RightBracket') # random light and ball position
      # num = ['One','Two','Three','Four','Five']
      # self.unrealcv.keyboard(num[random.randint(0,len(num)-1)])#  random material



