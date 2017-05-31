import gym # openai gym
from unrealcv_cmd import  UnrealCv # a lib for using unrealcv client command
import numpy as np
import math
import run_docker # a lib for run env in a docker container
'''
State :  color image
Action:  (linear velocity ,angle velocity)
Done :   Collision detected or get a target place
'''
class UnrealCvSimple(gym.Env):
    # init the Unreal Gym Environment
   def __init__(self,
                ENV_NAME = 'RealisticRendering'
   ):
     self.cam_id = 0
     # run virtual enrionment in docker container
     self.docker = run_docker.RunDocker()
     env_ip, env_dir = self.docker.start(ENV_NAME=ENV_NAME)
     # connect unrealcv client to server
     self.unrealcv = UnrealCv(self.cam_id, ip=env_ip, env=env_dir)
     self.startpose = self.unrealcv.get_pose()
     # ACTION: (linear velocity ,angle velocity)
     self.ACTION_LIST = [
             (20,  0),
             (20, 15),
             (20,-15),
             (20, 30),
             (20,-30),
     ]
     self.count_steps = 0
     self.max_steps = 100
     self.target_pos = ( -60,   0,   50)
     self.action_space = gym.spaces.Discrete(len(self.ACTION_LIST))
     state = self.unrealcv.read_image(self.cam_id, 'lit')
     self.observation_space = gym.spaces.Box(low=0, high=255, shape=state.shape)

   # update the environment step by step
   def _step(self, action = 0):
        (velocity, angle) = self.ACTION_LIST[action]
        self.count_steps += 1
        collision =  self.unrealcv.move(self.cam_id, angle, velocity)
        reward, done = self.reward(collision)
        state = self.unrealcv.read_image(self.cam_id, 'lit')

        # limit max step per episode
        if self.count_steps > self.max_steps:
            done = True
            print 'Reach Max Steps'

        return state, reward, done, {}

   # reset the environment
   def _reset(self, ):
       x,y,z,yaw = self.startpose
       self.unrealcv.set_position(self.cam_id, x, y, z)
       self.unrealcv.set_rotation(self.cam_id, 0, yaw, 0)
       state = self.unrealcv.read_image(self.cam_id, 'lit')
       self.count_steps = 0
       return  state

   # close docker while closing openai gym
   def _close(self):
       self.docker.close()

   # calcuate reward according to your task
   def reward(self,collision):
       done = False
       reward = - 0.01
       if collision:
            done = True
            reward = -1
            print 'Collision Detected!!'
       else:
            distance = self.cauculate_distance(self.target_pos, self.unrealcv.get_pos())
            if distance < 50:
                reward = 10
                done = True
                print ('Get Target Place!')
       return reward, done

   # calcuate the 2D distance between the target and camera
   def cauculate_distance(self,target,current):
       error = abs(np.array(target) - np.array(current))[:2]# only x and y
       distance = math.sqrt(sum(error * error))
       return distance