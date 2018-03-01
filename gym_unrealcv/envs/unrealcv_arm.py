import math
import os
import random
import time

import gym
import numpy as np
from gym import spaces
from gym_unrealcv.envs.robotarm.visualization import show_info
from gym_unrealcv.envs.utils import env_unreal
from gym_unrealcv.envs.robotarm.interaction import Robotarm

class UnrealCvRobotArm_base(gym.Env):
   def __init__(self,
                setting_file = 'search_rr_plant78.json',
                reset_type = 'keyboard',    # testpoint, waypoint,
                action_type = 'discrete',   # 'discrete', 'continuous'
                observation_type = 'color', # 'color', 'depth', 'rgbd' . 'measure'
                reward_type = 'move', # distance, move, move_distance
                docker = False,
                resolution=(640, 480),
                use_attach = False
                ):

     setting = self.load_env_setting(setting_file)
     self.docker = docker
     self.reset_type = reset_type
     self.use_attach = use_attach
     # start unreal env
     self.unreal = env_unreal.RunUnreal(ENV_BIN=setting['env_bin'])
     env_ip, env_port = self.unreal.start(docker,resolution)


     # connect UnrealCV
     self.unrealcv =Robotarm(cam_id=self.cam_id,
                             pose_range=self.pose_range,
                             port= env_port,
                             ip=env_ip,
                             targets= self.target_list,
                             env=self.unreal.path2env,
                             resolution= resolution)

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
     self.observation_space = self.unrealcv.define_observation(self.cam_id,observation_type)

     # define reward type
     # distance, bbox, bbox_distance,
     self.reward_type = reward_type
     self.rendering = False
     self.attached = False
     self.unrealcv.set_fov(60)




   def _step(self, action):
        info = dict(
            Collision=False,
            Done = False,
            Maxstep=False,
            Reward=0.0,
            Action = action,
            Bbox =None,
            ArmPose = [],
            GripPosition = [],
            Steps= self.count_steps+1,
            Target = self.target_list,
            TargetPose = self.target_pose,
            Color = None,
            Depth = None,
            QRPose = None
        )

        # take a action
        if self.action_type == 'discrete':
            arm_state = self.unrealcv.move_arm(self.discrete_actions[action])

        elif self.action_type == 'continuous':
            arm_state = self.unrealcv.move_arm(np.append(action, 0))



        self.count_steps += 1
        info['Done'] = False

        info['GripPosition'] = self.unrealcv.get_grip_position().tolist() # for reward
        info['ArmPose'] = self.unrealcv.arm['pose'].tolist()
        info['QRPose'] = self.unrealcv.get_QR_pose() # for observation
        if self.attached == False:
            info['TargetPose'] = self.unrealcv.get_obj_location(self.target_list[0])
            distance = self.get_distance(info['TargetPose'], info['GripPosition'])
        else:
            info['TargetPose'] = self.unrealcv.get_obj_location(self.target_list[1])
            info['TargetPose'][2] = 80
            distance = self.get_distance(info['TargetPose'], info['GripPosition'],2)
            if distance > 500:
                info['TargetPose'] = self.unrealcv.reset_obj(self.target_list[1],self.box_area)
                info['TargetPose'][2] = 80
                distance = self.get_distance(info['TargetPose'], info['GripPosition'],3)

        # reward function
        if distance < 15 and self.attached == True: #when moving object to target place
            info['Done'] = True
            info['Reward'] = 100
            # show detach ball
            '''
            self.unrealcv.detach_ball()
            self.attached = False
            time.sleep(2)
            '''

        elif self.attached == False and arm_state[6] == True: # reach target

            if self.use_attach:
                if self.unrealcv.attach_ball() and 'move' in self.reward_type:
                        info['Reward'] = 100
                        self.attached = True
                        time.sleep(1)
                        self.unrealcv.get_arm_pose()

                        info['TargetPose'] = self.unrealcv.get_obj_location(self.target_list[1])
                        info['TargetPose'][2] = 80
                        print ('move ball')
            else:
                info['Done'] = True
                info['Reward'] = 100

        elif arm_state[0] + arm_state[1]*~arm_state[2] + arm_state[3]*~arm_state[4] + arm_state[5] + arm_state[7] > 0 == False: # detect collision
            info['Collision'] = True
            info['Reward'] = -10
            info['Done'] = True

        elif arm_state[-1] : # reach pose limitation
            info['Reward'] = -1
            self.count_collision += 1
            if self.count_collision >= 5:
                info['Done'] = True
        else: # others
            info['Reward'] = -0.1
            if 'distance' in self.reward_type:
                distance_delt = self.distance_last - distance
                self.distance_last = distance
                info['Reward'] = distance_delt / 10.0

        # Get observation
        state = self.unrealcv.get_observation(self.cam_id, self.observation_type, info['TargetPose'], action)
        # bbox
        #object_mask = self.unrealcv.read_image(cam_id=self.cam_id, viewmode='object_mask')
        #self.bboxes = self.unrealcv.get_bboxes(object_mask=object_mask, objects=self.target_list)

        # limit the max steps of every episode
        if self.count_steps > self.max_steps:
           info['Done'] = True
           info['Maxstep'] = True
           #print 'Reach Max Steps'

        if self.rendering:
            show_info(info)
        return state, info['Reward'], info['Done'], info
   def _reset(self ):

       # set start position
       self.unrealcv.set_location(self.cam_id, self.camera_pose[0][:3])
       self.unrealcv.set_rotation(self.cam_id, self.camera_pose[0][-3:])


       # for reset point generation and selection
       if self.attached:
           self.unrealcv.detach_ball()
           self.attached = False

       if self.reset_type == 'keyboard':
           #self.unrealcv.reset_env_keyboard()
           self.unrealcv.set_arm_pose([0, 0, 0, 0, 0])
           self.unrealcv.set_material('Ball0', rgb=[1, 0.2, 0.2], prop=np.random.random(3))

           self.unrealcv.keyboard('RightBracket')  # random light
           time.sleep(1)
           #self.unrealcv.reset_obj(self.target_list[1], self.box_area)
           while True:
               self.unrealcv.keyboard('LeftBracket')  # random ball position
               if not self.unrealcv.check_inbox():
                   break





       #self.unrealcv.get_grip_position()
       self.target_pose = self.unrealcv.reset_obj(self.target_list[0], self.ball_area)
       #self.target_pose = np.array(self.unrealcv.get_obj_location(self.target_list[0]))
       state = self.unrealcv.get_observation(self.cam_id, self.observation_type, self.target_pose)

       self.count_steps = 0
       self.count_collision = 0


       self.distance_last = self.get_distance(self.target_pose, self.unrealcv.get_grip_position())
       self.unrealcv.set_arm_pose(self.unrealcv.get_arm_pose())
       self.unrealcv.empty_msgs_buffer()

       return state

   def _close(self):
       if self.docker:
           self.unreal.docker.close()


   def _get_action_size(self):
       return len(self.action)


   def get_distance(self,target,current,n=3):

       error = np.array(target) - np.array(current)# only x and y
       distance = np.linalg.norm(error[:n])
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
       self.box_area = setting['box_area']
       self.ball_area = setting['ball_area']

       return setting


   def get_settingpath(self, filename):
       import gym_unrealcv
       gympath = os.path.dirname(gym_unrealcv.__file__)
       return os.path.join(gympath, 'envs/setting', filename)
