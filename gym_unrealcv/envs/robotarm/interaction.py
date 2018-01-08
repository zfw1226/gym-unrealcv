from gym_unrealcv.envs.utils.unrealcv_basic import UnrealCv
import numpy as np
import time
from gym import spaces
import random
import re
class Robotarm(UnrealCv):
    def __init__(self, env, pose_range, cam_id = 0, port = 9000, targets = None,
                 ip = '127.0.0.1',resolution=(160,120)):
        self.arm = dict(
                pose = np.zeros(5),
                state= np.zeros(7), # ground, left, left_in, right, right_in, body, reach
                grip = np.zeros(3),
                high = np.array(pose_range['high']),
                low = np.array(pose_range['low']),
                flag_grip = False,
        )

        super(Robotarm, self).__init__(env=env, port = port,ip = ip , cam_id=cam_id,resolution=resolution)

        if targets == 'all':
            self.targets = self.get_objects()
            self.color_dict = self.build_color_dic(self.targets)
        elif targets is not None:
            self.targets = targets
            self.color_dict = self.build_color_dic(self.targets)


        self.bad_msgs = []
        self.good_msgs = []
    def message_handler(self,msg):
        print 'receive msg'

    def read_message(self):
        good_msgs = self.good_msgs
        bad_msgs = self.bad_msgs
        self.empty_msgs_buffer()
        return good_msgs, bad_msgs

    def empty_msgs_buffer(self):
        self.good_msgs = []
        self.bad_msgs = []

    def set_arm_pose(self,pose):
        self.arm['pose'] = np.array(pose)
        cmd = 'vexec armBP setpos {grip} {M3} {M2} {M1} {M0}'
        return self.client.request(cmd.format(M0=pose[0],M1=pose[1],M2=pose[2],M3=pose[3],grip=pose[4]))

    def move_arm(self,action):
        pose_tmp = self.arm['pose']+action
        out_max = pose_tmp > self.arm['high']
        out_min = pose_tmp < self.arm['low']
        #print pose_tmp,out_max,out_max
        if out_max.sum() + out_min.sum() == 0:
            limit = False
        else:
            limit = True
            pose_tmp = out_max * self.arm['high'] + out_min* self.arm['low'] + ~(out_min+out_max)*pose_tmp
            #print pose_tmp

        self.set_arm_pose(pose_tmp)
        state = self.get_arm_state()
        state.append(limit)
        #print state
        return state

    def get_arm_pose(self):
        cmd = 'vbp armBP getpos'
        result = self.client.request(cmd)
        result = result.split()
        pose = []
        for i in range(2,11,2):
            pose.append(float(result[i][1:-2]))
        pose.reverse()
        self.arm['pose'] = np.array(pose)
        #print pose
        return self.arm['pose']

    def get_arm_state(self):
        cmd = 'vbp armBP querysetpos'
        result = None
        while result is None:
            result = self.client.request(cmd)
        result = result.split()
        state = []
        for i in range(2,15,2):
            if result[i][1:5] == 'true':
                state.append(True)
            else:
                state.append(False)
        self.arm['state'] = state
        return state


    def get_grip_position(self):
        cmd = 'vbp armBP getgrip'
        result = None
        while result is None:
            result = self.client.request(cmd)
        result = result.split()
        position = []
        for i in range(2,7,2):
            position.append(float(result[i][1:-2]))
        self.arm['grip'] = np.array(position)
        return self.arm['grip']

    def define_observation(self,cam_id, observation_type):
        if observation_type == 'color':
            state = self.read_image(cam_id, 'lit','fast')
            observation_space = spaces.Box(low=0, high=255, shape=state.shape)
        elif observation_type == 'depth':
            state = self.read_depth(cam_id)
            observation_space = spaces.Box(low=0, high=100, shape=state.shape)
        elif observation_type == 'rgbd':
            state = self.get_rgbd(cam_id)
            s_high = state
            s_high[:, :, -1] = 100.0  # max_depth
            s_high[:, :, :-1] = 255  # max_rgb
            s_low = np.zeros(state.shape)
            observation_space = spaces.Box(low=s_low, high=s_high)
        elif observation_type == 'measured':
            s_high = [130,  60,  170, 50, 70,  200,  300, 360, 250, 400, 360]  # arm_pose, grip_position, target_position
            s_low = [-130, -90, -60, -50,  0, -400, -150, 0, -350, -150, 40]
            observation_space = spaces.Box(low=np.array(s_low), high=np.array(s_high))
        return observation_space

    def get_observation(self,cam_id, observation_type):
        if observation_type == 'color':
            self.img_color = state = self.read_image(cam_id, 'lit','fast')
        elif observation_type == 'depth':
            self.img_depth = state = self.read_depth(cam_id)
        elif observation_type == 'rgbd':
            self.img_color = self.read_image(cam_id, 'lit','fast')
            self.img_depth = self.read_depth(cam_id)
            state = np.append(self.img_color, self.img_depth, axis=2)
        elif observation_type == 'measured':
            arm_pose = self.arm['pose'].copy()
            self.target_pose = np.array(self.get_obj_location(self.targets[0]))
            state = np.concatenate((arm_pose, self.arm['grip'], self.target_pose))
            # [p0,p1,p2,p3,p4,g_x,g_y,g_z,g_r,g_y,g_p,t_x,t_y,t_z]
        return state

    def reset_env_keyboard(self):
        #self.keyboard('R')  # reset arm pose
        self.set_arm_pose([0,0,0,0,0])
        self.set_material('Ball0', rgb=[1,0.2,0.2], prop=np.random.random(3))
        #self.keyboard('LeftBracket')   # random ball position
        self.keyboard('RightBracket')  # random light
        time.sleep(1)
        self.keyboard('LeftBracket')  # random ball position

    def random_material(self):
        self.set_material('Ball0',rgb=np.random.random(3),prop=np.random.random(3))
        self.set_material('wall1',rgb=np.random.random(3),prop=np.random.random(3))
        self.set_material('wall2', rgb=np.random.random(3), prop=np.random.random(3))
        self.set_material('wall3', rgb=np.random.random(3), prop=np.random.random(3))
        self.set_material('wall4', rgb=np.random.random(3), prop=np.random.random(3))
        self.set_arm_material('yellow', rgb=np.random.random(3), prop=np.random.random(3))
        self.set_arm_material('black', rgb=np.random.random(3), prop=np.random.random(3))

    def attach_ball(self):
        return self.client.request('vbp armBP catch')

    def detach_ball(self):
        return self.client.request('vbp armBP loose')

    def set_material(self,target, rgb=(1,1,1), prop = (1,1,1)): # Ball0 wall1/2/3/4
        cmd = 'vbp {target} setmaterial {r} {g} {b} {metallic} {specular} {roughness}'
        return self.client.request(cmd.format(target = target, r=rgb[0],g=rgb[1], b=rgb[2], metallic=prop[0], specular=prop[1], roughness=prop[2]))

    def set_arm_material(self,target, rgb=(1,1,1), prop = (1,1,1)):# yellow black
        cmd = 'vbp armBP setarm{target} {r} {g} {b} {metallic} {specular} {roughness}'
        return self.client.request(cmd.format(target = target, r=rgb[0], g=rgb[1], b=rgb[2], metallic=prop[0], specular=prop[1], roughness=prop[2]))

