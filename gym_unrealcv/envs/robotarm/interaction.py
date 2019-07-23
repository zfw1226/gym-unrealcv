from gym_unrealcv.envs.utils.unrealcv_basic import UnrealCv
import numpy as np
import time
from gym import spaces
import random
import re
class Robotarm(UnrealCv):
    def __init__(self, env, pose_range, cam_id=0, port=9000, targets=None,
                 ip='127.0.0.1', resolution=(160, 120)):
        self.arm = dict(
                pose=np.zeros(5),
                state=np.zeros(8),  # ground, left, left_in, right, right_in, body, reach
                grip=np.zeros(3),
                high=np.array(pose_range['high']),
                low=np.array(pose_range['low']),
        )
        super(Robotarm, self).__init__(env=env, port=port, ip=ip, cam_id=cam_id, resolution=resolution)

        if targets == 'all':
            self.targets = self.get_objects()
            self.color_dict = self.build_color_dic(self.targets)
        elif targets is not None:
            self.targets = targets
            self.color_dict = self.build_color_dic(self.targets)
        self.msgs_buffer = []

    def message_handler(self, msg):
        # msg: 'Hit object'
        self.msgs_buffer.append(msg)

    def read_message(self):
        msgs = self.msgs_buffer
        self.empty_msgs_buffer()
        return msgs

    def empty_msgs_buffer(self):
        self.msgs_buffer = []

    def set_arm_pose(self, pose, mode='old'):
        self.arm['pose'] = np.array(pose)
        if mode == 'new':
            cmd = 'vset /arm/RobotArmActor_1/pose {M0} {M1} {M2} {M3} {grip}'
        elif mode == 'move':
            cmd = 'vset /arm/RobotArmActor_1/moveto {M0} {M1} {M2} {M3} {grip}'
        elif mode == 'old':
            cmd = 'vbp armBP setpos {grip} {M3} {M2} {M1} {M0}'
        return self.client.request(cmd.format(M0=pose[0], M1=pose[1], M2=pose[2],
                                              M3=pose[3], grip=pose[4]))

    def move_arm(self, action, mode='old'):
        pose_tmp = self.arm['pose']+action
        out_max = pose_tmp > self.arm['high']
        out_min = pose_tmp < self.arm['low']

        if out_max.sum() + out_min.sum() == 0:
            limit = False
        else:
            limit = True
            pose_tmp = out_max*self.arm['high'] + out_min*self.arm['low'] + ~(out_min+out_max)*pose_tmp
        self.set_arm_pose(pose_tmp, mode)

        if mode == 'old':
            state = self.get_arm_state()
            state.append(limit)
        else:
            self.arm['pose'] = pose_tmp
            state = limit
        return state

    def get_arm_pose(self, mode='old'):
        if mode == 'old':
            cmd = 'vbp armBP getpos'
        else:
            cmd = 'vget /arm/RobotArmActor_1/pose'
        result = None
        while result is None:
            result = self.client.request(cmd)
        result = result.split()
        if mode=='old':
            pose = []
            for i in range(2, 11, 2):
                pose.append(float(result[i][1:-2]))
            pose.reverse()
        else:
            pose = [float(i) for i in result]
        self.arm['pose'] = np.array(pose)
        return self.arm['pose']

    def get_tip_pose(self):
        cmd = 'vget /arm/RobotArmActor_1/tip_pose'
        result = None
        while result is None:
            result = self.client.request(cmd)
        pose = np.array([float(i) for i in result.split()])
        pose[1] = -pose[1]
        self.arm['grip'] = pose[:3]
        return pose

    def define_observation(self, cam_id, observation_type, setting, mode='fast'):
        if observation_type != 'Pose':
            state = self.get_observation(cam_id, observation_type, mode=mode)
        if observation_type == 'Color' or observation_type == 'CG':
            observation_space = spaces.Box(low=0, high=255, shape=state.shape, dtype=np.uint8)  # for gym>=0.10
        elif observation_type == 'Depth':
            observation_space = spaces.Box(low=0, high=100, shape=state.shape, dtype=np.float16)  # for gym>=0.10
        elif observation_type == 'Rgbd':
            s_high = state
            s_high[:, :, -1] = 100.0  # max_depth
            s_high[:, :, :-1] = 255  # max_rgb
            s_low = np.zeros(state.shape)
            observation_space = spaces.Box(low=s_low, high=s_high, dtype=np.float16)  # for gym>=0.10`
        elif observation_type == 'Pose':
            s_high = setting['pose_range']['high'] + setting['goal_range']['high'] + setting['continous_actions']['high']  # arm_pose, target_position, action
            s_low = setting['pose_range']['low'] + setting['goal_range']['low'] + setting['continous_actions']['low']
            observation_space = spaces.Box(low=np.array(s_low), high=np.array(s_high))
        return observation_space

    def get_observation(self, cam_id, observation_type, target_pose=np.zeros(3), action=np.zeros(4), mode='fast'):
        if observation_type == 'Color':
            self.img_color = state = self.read_image(cam_id, 'lit', mode)
        elif observation_type == 'Depth':
            self.img_depth = state = self.read_depth(cam_id)
        elif observation_type == 'Rgbd':
            self.img_color = self.read_image(cam_id, 'lit', mode)
            self.img_depth = self.read_depth(cam_id)
            state = np.append(self.img_color, self.img_depth, axis=2)
        elif observation_type == 'Pose':
            self.target_pose = np.array(target_pose)
            state = np.concatenate((self.arm['pose'], self.target_pose, action))
        return state

    def check_collision(self, obj='RobotArmActor_1'):
        'cmd : vget /arm/RobotArmActor_1/query collision'
        cmd = 'vget /arm/{obj}/query collision'
        res = self.client.request(cmd.format(obj=obj))
        if res == 'true':
            return True
        else:
            return False