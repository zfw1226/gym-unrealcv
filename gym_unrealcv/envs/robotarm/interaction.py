from gym_unrealcv.envs.utils.unrealcv_basic import UnrealCv
import numpy as np
import time
from gym import spaces
import random
class Robotarm(UnrealCv):
    def __init__(self, env, cam_id = 0, port = 9000, targets = None,
                 ip = '127.0.0.1',resolution=(160,120)):
        self.arm = dict(
                pose = np.zeros(5),
                flag_pose = False,
                grip = np.zeros(3),
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

        #filter for pose
        if 'Currentpose' in msg:
            pose_str = msg[12:].split()
            self.arm['pose'] = np.array(pose_str,dtype=np.float16)
            self.arm['flag_pose'] = True
        elif 'GripLocation' in msg:
            pose_str = msg[13:].split()
            self.arm['grip'] = np.array(pose_str, dtype=np.float16)
            self.arm['flag_grip'] = True
        elif msg == 'move':
            self.good_msgs.append(msg)
        else:
            self.bad_msgs.append(msg)

    def read_message(self):

        good_msgs = self.good_msgs
        bad_msgs = self.bad_msgs
        self.empty_msgs_buffer()
        return good_msgs, bad_msgs

    def empty_msgs_buffer(self):
        self.good_msgs = []
        self.bad_msgs = []

    def get_arm_pose(self):

        self.keyboard('C')
        start_time = time.time()
        while self.arm['flag_pose']==False:
            delt_time = time.time() - start_time
            if delt_time > 0.5:  # time out
                break
        self.arm['flag_pose'] = False
        return self.arm['pose']


    def get_grip_position(self):

        self.keyboard('F')
        start_time = time.time()
        while self.arm['flag_grip']==False:
            delt_time = time.time() - start_time
            if delt_time > 0.5:  # time out
                print 'time out'
                break
        self.arm['flag_grip'] = False
        return self.arm['grip']

    def define_observation(self,cam_id, observation_type):
        if observation_type == 'color':
            state = self.read_image(cam_id, 'lit')
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
            s_high = [85, 80, 90, 95, 120, 200, 300, 360, 250, 400, 360]  # arm_pose, grip_position, target_position
            s_low = [0, -90, -60, -55, -120, -400, -150, 0, -350, -150, 40]
            observation_space = spaces.Box(low=np.array(s_low), high=np.array(s_high))
        return observation_space

    def get_observation(self,cam_id, observation_type):
        if observation_type == 'color':
            self.img_color = state = self.read_image(cam_id, 'lit')
        elif observation_type == 'depth':
            self.img_depth = state = self.read_depth(cam_id)
        elif observation_type == 'rgbd':
            self.img_color = self.read_image(cam_id, 'lit')
            self.img_depth = self.read_depth(cam_id)
            state = np.append(self.img_color, self.img_depth, axis=2)
        elif observation_type == 'measured':
            self.message = []
            arm_pose = self.arm['pose'].copy()
            self.target_pose = np.array(self.get_obj_location(self.targets[0]))
            state = np.concatenate((arm_pose, self.arm['grip'], self.target_pose))
            # [p0,p1,p2,p3,p4,g_x,g_y,g_z,g_r,g_y,g_p,t_x,t_y,t_z]
        return state

    def reset_env_keyboard(self):
        self.keyboard('R')  # reset arm pose
        time.sleep(0.1)
        self.keyboard('LeftBracket')
        self.keyboard('RightBracket')  # random light and ball position
        #num = ['One', 'Two', 'Three', 'Four', 'Five']
        #self.keyboard(num[random.randint(0, len(num) - 1)])  # random material