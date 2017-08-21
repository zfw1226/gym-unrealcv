import gym
from gym import spaces
from unrealcv_cmd import  UnrealCv
import numpy as np
import time
import random
import math
import os
from operator import itemgetter
import env_unreal
import reward
import reset_point
from visualization import show_info_arm

class UnrealCvRobotArm_base(gym.Env):
   def __init__(self,
                setting_file = 'search_rr_plant78.json',
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
     env_ip = self.unreal.start(docker)

     # connect UnrealCV
     self.unrealcv = UnrealCv(cam_id=self.cam_id,
                              port= 9000,
                              ip=env_ip,
                              targets=None,
                              env=self.unreal.path2env)

    # define action type
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



   def _step(self, action , show = True):
        info = dict(
            Collision=False,
            Done = False,
            Maxstep=False,
            Reward=0.0,
            Action = action,
            Bbox =[],
            Pose = [],
            Steps = self.count_steps,
            Target = [],
            Color = None,
            Depth = None,
        )

        if self.action_type == 'discrete':
            duration = 0.2
            self.unrealcv.keyboard(self.discrete_actions[action], duration=duration)
            time.sleep(duration)

        self.count_steps += 1
        info['Done'] = False

        if self.observation_type == 'color':
            state = info['Color'] = self.unrealcv.read_image(self.cam_id, 'lit')
        elif self.observation_type == 'depth':
            state = info['Depth'] = self.unrealcv.read_depth(self.cam_id)
        elif self.observation_type == 'rgbd':
            info['Color'] = self.unrealcv.read_image(self.cam_id, 'lit')
            info['Depth'] = self.unrealcv.read_depth(self.cam_id)
            state = np.append(info['Color'], info['Depth'], axis=2)

        msg = self.unrealcv.read_message()
        if msg == None :  # get reward by distance
            info['Reward'] = -0.01

        elif msg == 'move':  # touch target
            info['Reward'] = 10
            info['Done'] = True
            print 'Move ball'

        else:  # collision
            info['Collision'] = True
            info['Reward'] = -1
            info['Done'] = False
            print 'Collision'

        # limit the max steps of every episode
        if self.count_steps > self.max_steps:
           info['Done'] = True
           info['Maxstep'] = True
           print 'Reach Max Steps'

        if show:
            show_info_arm(info)

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

       self.count_steps = 0

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
           print 'unknown type'

       #print setting
       self.cam_id = setting['cam_view_id']
       self.cam_arm_id = setting['cam_arm_id']
       #self.target_list = setting['targets']
       self.max_steps = setting['maxsteps']
       self.camera_pose = setting['camera_pose']
       self.discrete_actions = setting['discrete_actions']
       self.continous_actions = setting['continous_actions']

       return setting


   def get_settingpath(self, filename):
       import gym_unrealcv
       gympath = os.path.dirname(gym_unrealcv.__file__)
       return os.path.join(gympath, 'envs/setting', filename)


   def open_door(self):
       self.unrealcv.keyboard('RightMouseButton')
       time.sleep(2)
       self.unrealcv.keyboard('RightMouseButton') # close the door

   def reset_env_keyboard(self):
       self.unrealcv.keyboard('RightBracket') # random light and ball position
       num = ['One','Two','Three','Four','Five']
       self.unrealcv.keyboard(num[random.randint(0,len(num)-1)])#  random material
       self.unrealcv.keyboard('R') # reset arm pose

