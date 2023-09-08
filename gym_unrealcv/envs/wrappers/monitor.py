import gym
from gym import Wrapper
import time
import cv2

class DisplayWrapper(Wrapper):
    def __init__(self, env, dynamic_top_down=True, fix_camera=True, get_bbox=True):
        super().__init__(env)
        self.dynamic_top_down = dynamic_top_down
        self.fix_camera = fix_camera
        self.get_bbox = get_bbox

    def step(self, action):
        obs, reward, done, info = self.env.step(action) # take a step in the wrapped environment
        # set top_down camera
        env = self.env.unwrapped

        # for recording demo
        if self.get_bbox:
            mask = env.unrealcv.read_image(env.cam_list[env.tracker_id], 'object_mask', 'fast')
            mask, bbox = env.unrealcv.get_bbox(mask, env.player_list[env.target_id], normalize=False)
            self.show_bbox(env.img_show.copy(), bbox)
            info['bbox'] = bbox

        if self.dynamic_top_down:
            env.set_topview(info['Pose'], env.cam_id[0]) # set top_down camera

        return obs, reward, done, info # return the same results as the wrapped environment

    def reset(self, **kwargs):
        states = self.env.reset(**kwargs)
        env = self.env.unwrapped
        if self.fix_camera:
            center_pos = [(env.reset_area[0]+env.reset_area[1])/2, (env.reset_area[2]+env.reset_area[3])/2, 2000]
            env.set_topview(center_pos, env.cam_id[0])
        if self.get_bbox:
            self.bbox_init = []
            mask = env.unrealcv.read_image(env.cam_list[env.tracker_id], 'object_mask', 'fast')
            mask, bbox = env.unrealcv.get_bbox(mask, env.player_list[env.target_id], normalize=False)
            self.mask_percent = mask.sum()/(255 * env.resolution[0] * env.resolution[1])
            self.bbox_init.append(bbox)
        cv2.imshow('init', env.img_show)
        cv2.waitKey(1)
        return states # return the same results as the wrapped environment

    def show_bbox(self, img2disp, bbox):
        # im_disp = states[0][:, :, :3].copy()
        cv2.rectangle(img2disp, (int(bbox[0]), int(bbox[1])), (int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])), (0, 255, 0), 5)
        cv2.imshow('track_res', img2disp)
        cv2.waitKey(1)