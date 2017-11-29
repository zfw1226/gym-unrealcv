from gym_unrealcv.envs.utils.unrealcv_basic import UnrealCv
import numpy as np
import time
from gym import spaces
class Navigation(UnrealCv):
    def __init__(self, env, cam_id = 0, port = 9000,
                 ip = '127.0.0.1' , targets = None, resolution=(160,120)):

        super(Navigation, self).__init__(env=env, port = port,ip = ip , cam_id=cam_id,resolution=resolution)

        if targets == 'all':
            self.targets = self.get_objects()
            self.color_dict = self.build_color_dic(self.targets)
        elif targets is not None:
            self.targets = targets
            self.color_dict = self.build_color_dic(self.targets)

        self.img_color = np.zeros(1)
        self.img_depth = np.zeros(1)


    def get_observation(self,cam_id, observation_type):
        if observation_type == 'color':
            self.img_color = state = self.read_image(cam_id, 'lit')
        elif observation_type == 'depth':
            self.img_depth = state = self.read_depth(cam_id)
        elif observation_type == 'rgbd':
            self.img_color = self.read_image(cam_id, 'lit')
            self.img_depth = self.read_depth(cam_id)
            state = np.append(self.img_color, self.img_depth, axis=2)
        return state

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
        return observation_space
    def open_door(self):
        self.keyboard('RightMouseButton')
        time.sleep(2)
        self.keyboard('RightMouseButton')  # close the door
#nav = Navigation(env='test')