import gym
from unrealcv_cmd import  UnrealCv
import numpy as np
import time
import random
import math
from gym import spaces
import os
from operator import itemgetter
import env_unreal

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
                setting_file = 'search_rr_plant78.yaml',
                test = True,
                action_type = 'discrete',  # 'discrete', 'continuous'
                observation_type = 'rgbd', # 'color', 'depth', 'rgbd'
                reward_type = 'bbox', # distance, bbox, bbox_distance,
                docker = False
                ):

     setting = self.load_env_setting(setting_file)
     self.test = test
     self.docker = docker

     # start unreal env
     self.unreal = env_unreal.RunUnreal(ENV_BIN=setting['env_bin'])
     env_ip = self.unreal.start(docker)

     # connect UnrealCV
     self.unrealcv = UnrealCv(cam_id=self.cam_id,
                              port= 9000,
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
     self.targets_pos = self.unrealcv.get_objects_pos(self.target_list)

    # define observation
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
         s_high = s_low = state
         s_high[:,:,-1] = 10.0
         s_high[:,:,:-1] = 255
         s_low[:,:,:] = 0
         self.observation_space = spaces.Box(low=s_low, high=s_high)

     # define reward
     self.reward_type = reward_type


     # set start position
     self.trigger_count  = 0
     current_pose = self.unrealcv.get_pose()
     current_pose[2] = self.height
     self.unrealcv.set_position(self.cam_id,current_pose[0],current_pose[1],current_pose[2])
     self.distance_last, self.target_last = self.select_target_by_distance(current_pose, self.targets_pos)

     # for reset point generation
     self.waypoints = []
     self.trajectory = []
     self.new_waypoint(current_pose,1000)
     self.collisionpoints = []
     self.start_id = 0
     self.yaw_id = 0

     self.collision = False
   def _step(self, action , show = False):
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
            Direction = self.distance_last,
            Waypoints = self.waypoints,
            Testpoints = self.testpoints,
            Color = None,
            Depth = None,
        )


        if self.action_type == 'discrete':
            (velocity, angle, info['Trigger']) = self.discrete_actions[action]
        else:
            (velocity, angle, info['Trigger']) = action
        self.count_steps += 1
        info['Done'] = False

        # the robot think that it found the target object,the episode is done
        # and get a reward by bounding box size
        # only three times false trigger allowed in every episode
        if info['Trigger'] > self.trigger_th :
            # get observation
            if self.observation_type == 'color':
                state = info['Color'] = self.unrealcv.read_image(self.cam_id, 'lit')
            elif self.observation_type == 'depth':
                state = info['Depth'] = self.unrealcv.read_depth(self.cam_id)
            elif self.observation_type == 'rgbd':
                info['Color'] = self.unrealcv.read_image(self.cam_id, 'lit')
                info['Depth'] = self.unrealcv.read_depth(self.cam_id)
                state = np.append(info['Color'], info['Depth'], axis=2)


            # get reward
            info['Pose'] = self.unrealcv.get_pose()
            self.trigger_count += 1

            if self.reward_type == 'bbox_distance' or self.reward_type == 'bbox':
                info['Reward'],info['Bbox'] = self.reward_bbox()
            else:
                info['Reward'] = 0

            if info['Reward'] > 0 or self.trigger_count > 3:
                info['Done'] = True
                if info['Reward'] > 0:
                    self.waypoints[self.start_id]['successed'] += 1
                    self.waypoints[self.start_id]['steps2target'].append(self.count_steps)

                print 'Trigger Terminal!'
        # if collision occurs, the episode is done and reward is -1
        else :
            # take action
            self.collision = info['Collision'] = self.unrealcv.move(self.cam_id, angle, velocity)
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
            info['Pose'] = self.unrealcv.get_pose()
            distance, self.target_id = self.select_target_by_distance(info['Pose'][:3],self.targets_pos)
            info['Target'] = self.targets_pos[self.target_id]
            if self.reward_type=='distance' or self.reward_type == 'bbox_distance':
                info['Reward'] = self.reward_distance(distance, self.target_id)
            else:
                info['Reward'] = 0

            info['Direction'] = self.get_direction (info['Pose'],self.targets_pos[self.target_id])

            if info['Collision']:
                info['Reward'] = -1
                info['Done'] = True
                self.collisionpoints.append(info['Pose'])
                print ('Collision!!')

        # limit the max steps of every episode
        if self.count_steps > self.max_steps:
           info['Done'] = True
           info['Maxstep'] = True
           print 'Reach Max Steps'

        # save the trajectory
        self.trajectory.append(info['Pose'])
        info['Trajectory'] = self.trajectory

        if show:
            self.unrealcv.show_img(info['Color'], 'state')

        return state, info['Reward'], info['Done'], info
   def _reset(self, ):
       # select a starting point
       if self.test:
           current_pose = self.reset_from_testpoint()
       else:
           if len(self.trajectory) > 5:
               self.update_waypoint()
           current_pose = self.reset_from_waypoint()

       if self.observation_type == 'color':
           state = self.unrealcv.read_image(self.cam_id, 'lit')
       elif self.observation_type == 'depth':
           state = self.unrealcv.read_depth(self.cam_id)
       elif self.observation_type == 'rgbd':
           state = self.unrealcv.get_rgbd(self.cam_id)


       self.trajectory = []
       self.trajectory.append(current_pose)
       self.trigger_count = 0
       self.count_steps = 0
       self.distance_last, self.target_last = self.select_target_by_distance(current_pose, self.targets_pos)

       return state

   def _close(self):
       if self.docker:
           self.unreal.docker.close()

       #sys.exit()


   def _get_action_size(self):
       return len(self.action)

# functions for starting point module
   def reset_from_testpoint(self):
       x,y = self.testpoints[self.start_id]
       z = self.height
       yaw = self.yaw_id * 45
       self.unrealcv.set_position(self.cam_id, x, y, z)
       self.unrealcv.set_rotation(self.cam_id, 0, yaw, 0)
       self.yaw_id += 1
       if self.yaw_id >=8:
           self.start_id = (self.start_id + 1) % len(self.testpoints)
           self.yaw_id = 0
       return [x,y,z,yaw]

   def reset_from_waypoint(self):
       # reset from waypoints generated in exploration
       x, y, z, yaw = self.select_waypoint_times()
       yaw = random.randint(0, 360)
       self.unrealcv.set_position(self.cam_id,x,y,z)
       self.unrealcv.set_rotation(self.cam_id, 0, yaw, 0)
       return [x,y,z,yaw]

   def select_waypoint_distance(self,currentpose):
       # select the farthest point in history
       dis = dict()
       i = 0
       for wp in self.waypoints:
           dis[i]= self.get_distance(currentpose,wp['pose'])
           i += 1
       dis_list = sorted(dis.items(), key=lambda item: item[1], reverse=True)
       # random sample the pos
       start_id = random.randint(0,(len(dis_list) - 1)/2)
       startpoint = self.waypoints[dis_list[start_id][0]]
       return startpoint

   def select_waypoint_random(self):
       self.start_id = random.randint(0, (len(self.waypoints) - 1) )
       startpoint = self.waypoints[self.start_id]['pose']
       self.waypoints[self.start_id]['selected'] += 1
       return startpoint

   def select_waypoint_times(self):

       self.waypoints = sorted(self.waypoints,key=itemgetter('selected'))
       self.start_id = random.randint(0, (len(self.waypoints) - 1)/3 )
       self.waypoints[self.start_id]['selected'] += 1

       return self.waypoints[self.start_id]['pose']

   def update_waypoint(self):
       #delete point close to collision point
       if len(self.collisionpoints)>1:

           if self.collision:
               self.update_dis2collision(self.trajectory[-1])
               self.sollision = False

           for P in self.trajectory:

               dis2waypoint,waypoint_id, dis2others = self.get_dis2waypoints(P)
               dis2collision = self.get_dis2collision(P)

               # update waypint
               if dis2waypoint < self.waypoint_th/4 and dis2collision > self.waypoints[waypoint_id]['dis2collision'] and dis2others>self.waypoint_th :
                   self.waypoints[waypoint_id]['pose'] = P
                   self.waypoints[waypoint_id]['dis2collision'] = dis2collision
                   print 'update waypoint'

               if dis2waypoint > self.waypoint_th and dis2collision > self.collision_th:
                   self.new_waypoint(P,dis2collision)
                   print 'add new waypoint'

       return len(self.waypoints)

   def get_dis2collision(self,pose):
       dis2collision = 1000
       for C in self.collisionpoints:
           dis2collision = min(dis2collision, self.get_distance(pose, C))
       return dis2collision

   def new_waypoint(self,pose,dis2collision):
       waypoint = dict()
       waypoint['pose']=pose
       waypoint['successed'] = 0
       waypoint['selected'] = 0
       waypoint['dis2collision'] = dis2collision
       waypoint['steps2target'] = []
       self.waypoints.append(waypoint)
       return waypoint

   def get_dis2waypoints(self,pose):
       dis2waypoints = []
       for W in self.waypoints:
           dis2waypoints.append(self.get_distance(pose,W['pose']))
       dis2waypoints = np.array(dis2waypoints)
       arg = np.argsort(dis2waypoints)

       id_min = arg[0]
       dis_min = dis2waypoints[id_min]
       if len(dis2waypoints) > 1:
           dis_other = dis2waypoints[arg[1]]
       else:
           dis_other = dis_min
       return dis_min, id_min, dis_other

   def update_dis2collision(self,C_point):
       # update dis2collision of every waypoint when detect a new collision point
       for i in range(len(self.waypoints)):
           distance = self.get_distance(self.waypoints[i]['pose'],C_point)
           self.waypoints[i]['dis2collision'] = min(self.waypoints[i]['dis2collision'],distance)


   def reward_bbox(self):

       object_mask = self.unrealcv.read_image(self.cam_id, 'object_mask')

       boxes = self.unrealcv.get_bboxes(object_mask,self.target_list)
       reward = 0
       for box in boxes:
           reward += self.get_bbox_reward(box)

       if reward > self.reward_th:
            reward = min(reward * self.reward_factor, 10)
            print ('Get ideal Target!!!')
       elif reward == 0:
           reward = -1
           print ('Get Nothing')
       else:
           reward = 0
           print ('Get small Target!!!')

       return reward,boxes

   def get_bbox_reward(self,box):
       (xmin,ymin),(xmax,ymax) = box
       boxsize = (ymax - ymin) * (xmax - xmin)
       x_c = (xmax + xmin) / 2.0
       x_bias = x_c - 0.5
       discount = max(0, 1 - x_bias ** 2)
       reward = discount * boxsize
       return reward


   def get_distance(self,target,current):

       error = abs(np.array(target)[:2] - np.array(current)[:2])# only x and y
       distance = math.sqrt(sum(error * error))
       return distance

   def select_target_by_distance(self,current_pos, targets_pos):
       # find the nearest target, return distance and targetid
       distances = []
       for target_pos in targets_pos:
           distances.append(self.get_distance(target_pos, current_pos))
       distances = np.array(distances)
       distance_now = distances.min()
       target_id = distances.argmin()

       return distance_now,target_id

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
   def reward_distance(self,distance_now,target_id):

       if target_id == self.target_last:
           reward = (self.distance_last - distance_now) / 100.0
       else:
           reward = 0
       self.distance_last = distance_now
       self.target_last = target_id
       return reward

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


       print setting
       self.cam_id = setting['cam_id']
       self.target_list = setting['targets']
       self.max_steps = setting['maxsteps']
       self.reward_th = setting['reward_th']
       self.trigger_th = setting['trigger_th']
       self.reward_factor = setting['reward_factor']
       self.height = setting['height']
       self.testpoints = setting['start_xy']
       self.collision_th = setting['collision_th']
       self.waypoint_th = setting['waypoint_th']
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