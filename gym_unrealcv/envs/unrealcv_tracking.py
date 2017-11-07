import math
import os
import time

import gym
import numpy as np
from gym import spaces
from gym_unrealcv.envs.tracking import reward
from gym_unrealcv.envs.tracking.visualization import show_info
from gym_unrealcv.envs.utils import env_unreal
from gym_unrealcv.envs.utils.unrealcv_cmd import UnrealCv
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
     self.unrealcv = UnrealCv(cam_id=self.cam_id,
                              port= env_port,
                              ip=env_ip,
                              targets=self.target_list,
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
     self.target_pos = self.unrealcv.get_object_pos(self.target_list[0])
     time.sleep(0.5 + 0.3*random.random())
     cam_pos = self.target_pos
     self.unrealcv.set_position(self.cam_id,cam_pos[0],cam_pos[1],cam_pos[2])
     self.target_pos = self.unrealcv.get_object_pos(self.target_list[0])
     yaw = self.get_direction(self.target_pos, cam_pos)
     self.unrealcv.set_rotation(self.cam_id,self.roll,yaw,self.pitch)

     self.trajectory = []

     self.rendering = False

     self.objects_env = ['road_A_55', 'road_B_58', 'road_B2_61', 'road_B3_67', 'road_B4_70', 'road_B5', 'flg_tree_01_74', 'flg_tree_02_80', 'flg_tree_03_83', 'flg_tree_89', 'flg_tree_92', 'flg_tree_95', 'flg_tree_98', 'flg_tree_101', 'flg_tree_104', 'flg_tree_107', 'flg_bush_02_114', 'flg_bush_01_117', 'flg_bush_120', 'flg_fern_123', 'flg_fern2_127', 'prp_trafficLight_27', 'flg_bush_31', 'flg_bush_34', 'flg_fern3', 'flg_fern4', 'flg_tree_39', 'flg_tree_10', 'flg_tree_11', 'gardenPlanter_A_51', 'gardenPlanter_B_55', 'flg_tree_58', 'flg_bush_64', 'flg_bush_67', 'flg_bush_70', 'flg_tree_76', 'flg_tree_79', 'gardenPlanter_A2', 'flg_tree_15', 'flg_tree_16', 'flg_tree_17', 'flg_tree_18', 'flg_tree_19', 'flg_groundLeaves_91', 'StoreSign_4_mdl_94', 'StoreSign_2_mdl_97', 'Sign_4_mdl_128', 'StoreSign_1_mdl_131', 'StoreSign_3_mdl_137', 'Building_9_6', 'Building_7_17', 'Building_6_20', 'Building_9_32', 'Building_6_38', 'building_1_01_47', 'Building_2_01_50', 'Building_3_53', 'Building_6_56', 'Building_4_02_59', 'Building_8_01_74', 'Building_4_01_77', 'building_1_86', 'prp_garbageCan_5', 'prp_bench_14', 'prp_bench2', 'prp_bench3', 'prp_bench4', 'Floor2', 'prp_tableUmbrella3', 'prp_chair5', 'prp_table3', 'prp_chair6', 'prp_tableUmbrella4', 'prp_chair7', 'prp_table4', 'prp_chair8', 'prp_tableUmbrella5', 'prp_chair9', 'prp_table5', 'prp_chair10', 'prp_tableUmbrella6', 'prp_chair11', 'prp_table6', 'prp_chair12', 'prp_tableUmbrella7', 'prp_chair13', 'prp_table7', 'prp_chair14', 'prp_tableUmbrella8', 'prp_chair15', 'prp_table8', 'prp_chair16', 'prp_tableUmbrella9', 'prp_chair17', 'prp_table9', 'prp_chair18', 'prp_tableUmbrella10', 'prp_chair19', 'prp_table10', 'prp_chair20', 'prp_tableUmbrella11', 'prp_chair21', 'prp_table11', 'prp_chair22', 'prp_tableUmbrella12', 'prp_chair23', 'prp_table12', 'prp_chair24', 'prp_tableUmbrella13', 'prp_chair25', 'prp_table13', 'prp_chair26']

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
            #keyboard
            '''
            if self.discrete_actions[action] != "Stop":
                self.unrealcv.keyboard(self.discrete_actions[action], duration=0.1)
            '''
            # linear
            (velocity, angle) = self.discrete_actions[action]
        else:
            (velocity, angle) = action

        info['Collision'] = self.unrealcv.move(self.cam_id, angle, velocity)

        self.count_steps += 1

        info['Pose'] = self.unrealcv.get_pose()
        #print info['Pose']
        self.target_pos = self.unrealcv.get_object_pos(self.target_list[0])
        #print self.target_pos
        info['Direction'] = self.get_direction(self.target_pos, info['Pose'][:3]) - info['Pose'][-1]
        if info['Direction'] < -180:
            info['Direction'] += 360
        elif info['Direction'] > 180:
            info['Direction'] -= 360


        info['Distance'] = self.get_distance(self.target_pos,info['Pose'][:3])


        #print info['Distance'],info['Direction']

        # update observation
        if self.observation_type == 'color':
            state = info['Color'] = self.unrealcv.read_image(self.cam_id, 'lit')
        elif self.observation_type == 'depth':
            state = info['Depth'] = self.unrealcv.read_depth(self.cam_id)
        elif self.observation_type == 'rgbd':
            info['Color'] = self.unrealcv.read_image(self.cam_id, 'lit')
            info['Depth'] = self.unrealcv.read_depth(self.cam_id)
            state = np.append(info['Color'], info['Depth'], axis=2)



        # get reward

        if info['Distance'] > self.max_distance or abs(info['Direction'])> self.max_direction:
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

        return state, info['Reward'], info['Done'], info
   def _reset(self, ):
       if 'random' in self.reset_type:
           num_update = 3
           to_hiden = random.sample(self.show_list,num_update)
           #print to_hiden
           for x in to_hiden:
               self.show_list.remove(x)
               self.hiden_list.append(x)
           self.unrealcv.hide_objects(to_hiden)

           to_show = random.sample(self.hiden_list[:-num_update],num_update)
           #print to_show
           for x in to_show:
               self.hiden_list.remove(x)
               self.show_list.append(x)
           self.unrealcv.show_objects(to_show)






       self.target_pos = self.unrealcv.get_object_pos(self.target_list[0])
       time.sleep(0.5 + 0.3*random.random())
       cam_pos = self.target_pos
       self.unrealcv.set_position(self.cam_id, cam_pos[0], cam_pos[1], cam_pos[2])
       self.target_pos = self.unrealcv.get_object_pos(self.target_list[0])
       yaw = self.get_direction(self.target_pos, cam_pos)
       self.unrealcv.set_rotation(self.cam_id, self.roll, yaw, self.pitch)
       current_pose = self.unrealcv.get_pose()


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

       #sys.exit()


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

       return setting


   def get_settingpath(self, filename):
       import gym_unrealcv
       gympath = os.path.dirname(gym_unrealcv.__file__)
       return os.path.join(gympath, 'envs/setting', filename)
