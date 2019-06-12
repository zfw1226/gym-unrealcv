from gym_unrealcv.envs.utils.unrealcv_basic import UnrealCv
import numpy as np
import time
from gym import spaces
import gym
import distutils.version


class Navigation(UnrealCv):
    def __init__(self, env, cam_id=0, port=9000,
                 ip='127.0.0.1', targets=None, resolution=(160, 120)):
        super(Navigation, self).__init__(env=env, port=port, ip=ip, cam_id=cam_id, resolution=resolution)

        if targets == 'all':
            self.targets = self.get_objects()
            self.color_dict = self.build_color_dic(self.targets)
        elif targets is not None:
            self.targets = targets
            self.color_dict = self.build_color_dic(self.targets)

        self.img_color = np.zeros(1)
        self.img_depth = np.zeros(1)

        self.use_gym_10_api = distutils.version.LooseVersion(gym.__version__) >= distutils.version.LooseVersion('0.10.0')

    def get_observation(self, cam_id, observation_type, mode='direct'):
        if observation_type == 'Color':
            self.img_color = state = self.read_image(cam_id, 'lit', mode)
        elif observation_type == 'Depth':
            self.img_depth = state = self.read_depth(cam_id)
        elif observation_type == 'Rgbd':
            self.img_color = self.read_image(cam_id, 'lit', mode)
            self.img_depth = self.read_depth(cam_id)
            state = np.append(self.img_color, self.img_depth, axis=2)
        elif observation_type == 'CG':
            self.img_color = self.read_image(cam_id, 'lit', mode)
            self.img_gray = self.img_color.mean(2)
            self.img_gray = np.expand_dims(self.img_gray, -1)
            state = np.concatenate((self.img_color, self.img_gray), axis=2)
        return state

    def define_observation(self, cam_id, observation_type, mode='direct'):
        state = self.get_observation(cam_id, observation_type, mode)
        if observation_type == 'Color' or observation_type == 'CG':
            if self.use_gym_10_api:
                observation_space = spaces.Box(low=0, high=255, shape=state.shape, dtype=np.uint8)  # for gym>=0.10
            else:
                observation_space = spaces.Box(low=0, high=255, shape=state.shape)

        elif observation_type == 'Depth':
            if self.use_gym_10_api:
                observation_space = spaces.Box(low=0, high=100, shape=state.shape, dtype=np.float16)  # for gym>=0.10
            else:
                observation_space = spaces.Box(low=0, high=100, shape=state.shape)

        elif observation_type == 'Rgbd':
            s_high = state
            s_high[:, :, -1] = 100.0  # max_depth
            s_high[:, :, :-1] = 255  # max_rgb
            s_low = np.zeros(state.shape)
            if self.use_gym_10_api:
                observation_space = spaces.Box(low=s_low, high=s_high, dtype=np.float16)  # for gym>=0.10
            else:
                observation_space = spaces.Box(low=s_low, high=s_high)

        return observation_space

    def open_door(self):
        self.keyboard('RightMouseButton')
        time.sleep(2)
        self.keyboard('RightMouseButton')  # close the door

    def set_texture(self, target, color=(1, 1, 1), param=(0, 0, 0), picpath=None, tiling=1, e_num=0): #[r, g, b, meta, spec, rough, tiling, picpath]
        param = param / param.max()
        # color = color / color.max()
        cmd = 'vbp {target} set_mat {e_num} {r} {g} {b} {meta} {spec} {rough} {tiling} {picpath}'
        res = self.client.request(cmd.format(target=target, e_num=e_num, r=color[0], g=color[1], b=color[2],
                               meta=param[0], spec=param[1], rough=param[2], tiling=tiling,
                               picpath=picpath))

    def set_light(self, target, direction, intensity, color): # param num out of range
        cmd = 'vbp {target} set_light {row} {yaw} {pitch} {intensity} {r} {g} {b}'
        color = color/color.max()
        res = self.client.request(cmd.format(target=target, row=direction[0], yaw=direction[1],
                                             pitch=direction[2], intensity=intensity,
                                             r=color[0], g=color[1], b=color[2]))

    def set_skylight(self, target, color, intensity ): # param num out of range
        cmd = 'vbp {target} set_light {r} {g} {b} {intensity} '
        res = self.client.request(cmd.format(target=target, intensity=intensity,
                                             r=color[0], g=color[1], b=color[2]))

    def get_pose(self,cam_id, type='hard'):  # pose = [x, y, z, roll, yaw, pitch]
        if type == 'soft':
            pose = self.cam[cam_id]['location']
            pose.extend(self.cam[cam_id]['rotation'])
            return pose

        if type == 'hard':
            self.cam[cam_id]['location'] = self.get_location(cam_id)
            self.cam[cam_id]['rotation'] = self.get_rotation(cam_id)
            pose = self.cam[cam_id]['location'] + self.cam[cam_id]['rotation']
            return pose
