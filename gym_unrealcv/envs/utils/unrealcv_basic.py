import unrealcv
import cv2
import numpy as np
import math
import time
import os
import re
from io import BytesIO
import PIL.Image


class UnrealCv(object):
    def __init__(self, port, ip, env, cam_id, resolution):

        if ip == '127.0.0.1':
            self.docker = False
        else:
            self.docker = True
        self.client = unrealcv.Client((ip, port))

        self.envdir = env
        self.ip = ip
        print (self.ip)
        self.cam = dict()
        for i in range(6):
            self.cam[i] = dict(
                 location=[0, 0, 0],
                 rotation=[0, 0, 0],
            )

        self.init_unrealcv(cam_id, resolution)
        self.pitch = 0
        self.resolution = resolution
        self.img_color = None
        self.img_depth = None

    def init_unrealcv(self, cam_id, resolution=(320, 240)):
        self.client.connect()
        self.check_connection()
        self.client.request('vrun setres {w}x{h}w'.format(w=resolution[0], h=resolution[1]))
        self.client.request('vrun sg.ShadowQuality 0')
        self.client.request('vrun sg.TextureQuality 0')
        self.client.request('vrun sg.EffectsQuality 0')
        # self.client.request('vrun r.ScreenPercentage 10')
        # self.client.request('vrun t.maxFPS 100')
        time.sleep(1)
        self.get_rotation(cam_id, 'hard')
        self.get_location(cam_id, 'hard')
        self.client.message_handler = self.message_handler

    def message_handler(self, message):
        msg = message

    def check_connection(self):
        while self.client.isconnected() is False:
            print ('UnrealCV server is not running. Please try again')
            self.client.connect()

    def show_img(self, img, title="raw_img"):
        cv2.imshow(title, img)
        cv2.waitKey(3)

    def get_objects(self):
        objects = self.client.request('vget /objects')
        objects = objects.split()
        return objects

    def read_image(self, cam_id, viewmode, mode='direct'):
            # cam_id:0 1 2 ...
            # viewmode:lit,  =normal, depth, object_mask
            # mode: direct, file
            if mode == 'direct':
                cmd = 'vget /camera/{cam_id}/{viewmode} png'
                res = None
                while res is None:
                    res = self.client.request(cmd.format(cam_id=cam_id, viewmode=viewmode))
                image_rgb = self.decode_png(res)
                image_rgb = image_rgb[:, :, :-1]  # delete alpha channel
                image = image_rgb[:, :, ::-1]  # transpose channel order

            elif mode == 'file':
                cmd = 'vget /camera/{cam_id}/{viewmode} {viewmode}{ip}.png'
                if self.docker:
                    img_dirs_docker = self.client.request(cmd.format(cam_id=cam_id, viewmode=viewmode,ip=self.ip))
                    img_dirs = self.envdir + img_dirs_docker[7:]
                else :
                    img_dirs = self.client.request(cmd.format(cam_id=cam_id, viewmode=viewmode,ip=self.ip))
                image = cv2.imread(img_dirs)
            elif mode == 'fast':
                cmd = 'vget /camera/{cam_id}/{viewmode} bmp'
                res = None
                while res is None:
                    res = self.client.request(cmd.format(cam_id=cam_id, viewmode=viewmode))
                image_rgba = self.decode_bmp(res)
                image = image_rgba[:, :, :-1]  # delete alpha channel
            return image

    def read_depth(self, cam_id):

        cmd = 'vget /camera/{cam_id}/depth npy'
        res = self.client.request(cmd.format(cam_id=cam_id))
        depth = np.fromstring(res, np.float32)
        depth = depth[-self.resolution[1] * self.resolution[0]:]
        depth = depth.reshape(self.resolution[1], self.resolution[0],1)
        depth = depth/depth.max()
        #cv2.imshow('depth',depth/depth.max())
        #cv2.waitKey(10)
        return depth

    def decode_png(self, res):
        img = PIL.Image.open(BytesIO(res))
        return np.asarray(img)

    def decode_bmp(self, res, channel=4):
        img = np.fromstring(res, dtype=np.uint8)
        img=img[-self.resolution[1]*self.resolution[0]*channel:]
        img=img.reshape(self.resolution[1], self.resolution[0], channel)

        return img

    def convert2planedepth(self, PointDepth, f=320):
        H = PointDepth.shape[0]
        W = PointDepth.shape[1]
        i_c = np.float(H) / 2 - 1
        j_c = np.float(W) / 2 - 1
        columns, rows = np.meshgrid(np.linspace(0, W - 1, num=W), np.linspace(0, H - 1, num=H))
        DistanceFromCenter = ((rows - i_c) ** 2 + (columns - j_c) ** 2) ** 0.5
        PlaneDepth = PointDepth / (1 + (DistanceFromCenter / f) ** 2) ** 0.5
        return PlaneDepth

    def get_rgbd(self, cam_id, mode):
        rgb = self.read_image(cam_id, 'lit', mode)
        d = self.read_depth(cam_id)
        rgbd = np.append(rgb, d, axis=2)
        return rgbd

    def set_pose(self, cam_id, pose):  # pose = [x, y, z, roll, yaw, pitch]
        cmd = 'vset /camera/{cam_id}/pose {x} {y} {z} {pitch} {yaw} {roll}'
        self.client.request(cmd.format(cam_id=cam_id, x=pose[0], y=pose[1], z=pose[2], roll=pose[3], yaw=pose[4], pitch=pose[5]))
        self.cam[cam_id]['location'] = pose[:3]
        self.cam[cam_id]['rotation'] = pose[-3:]

    def get_pose(self, cam_id, mode='hard'):  # pose = [x, y, z, roll, yaw, pitch]
        if mode == 'soft':
            pose = self.cam[cam_id]['location']
            pose.extend(self.cam[cam_id]['rotation'])
            return pose

        if mode == 'hard':
            cmd = 'vget /camera/{cam_id}/pose'
            pose = None
            while pose is None:
                pose = self.client.request(cmd.format(cam_id=cam_id))
            pose = [float(i) for i in pose.split()]
            self.cam[cam_id]['location'] = pose[:3]
            self.cam[cam_id]['rotation'] = pose[-3:]
            return pose

    def set_location(self,cam_id, loc):  # loc=[x,y,z]
        cmd = 'vset /camera/{cam_id}/location {x} {y} {z}'
        self.client.request(cmd.format(cam_id=cam_id, x=loc[0], y=loc[1], z=loc[2]))
        self.cam[cam_id]['location'] = loc

    def get_location(self,cam_id, mode='hard'):
        if mode == 'soft':
            return self.cam[cam_id]['location']
        if mode == 'hard':
            cmd = 'vget /camera/{cam_id}/location'
            location = None
            while location is None:
                location = self.client.request(cmd.format(cam_id=cam_id))
            self.cam[cam_id]['location'] = [float(i) for i in location.split()]
            return self.cam[cam_id]['location']

    def set_rotation(self,cam_id, rot):  # rot = [roll, yaw, pitch]
        cmd = 'vset /camera/{cam_id}/rotation {pitch} {yaw} {roll}'
        self.client.request(cmd.format(cam_id=cam_id, roll=rot[0], yaw=rot[1], pitch=rot[2]))
        self.cam[cam_id]['rotation'] = rot

    def get_rotation(self, cam_id, mode='hard'):
        if mode == 'soft':
            return self.cam[cam_id]['rotation']
        if mode == 'hard':
            cmd = 'vget /camera/{cam_id}/rotation'
            rotation = None
            while rotation is None:
                rotation = self.client.request(cmd.format(cam_id=cam_id))
            rotation = [float(i) for i in rotation.split()]
            rotation.reverse()
            self.cam[cam_id]['rotation'] = rotation
            return self.cam[cam_id]['rotation']

    def moveto(self, cam_id, loc):
        cmd = 'vset /camera/{cam_id}/moveto {x} {y} {z}'
        self.client.request(cmd.format(cam_id=cam_id, x=loc[0], y=loc[1], z=loc[2]))

    def move_2d(self, cam_id, angle, length, height=0, pitch=0):

        yaw_exp = (self.cam[cam_id]['rotation'][1] + angle) % 360
        pitch_exp = (self.cam[cam_id]['rotation'][2] + pitch) % 360
        delt_x = length * math.cos(yaw_exp / 180.0 * math.pi)
        delt_y = length * math.sin(yaw_exp / 180.0 * math.pi)

        location_now = self.cam[cam_id]['location']
        location_exp = [location_now[0] + delt_x, location_now[1]+delt_y, location_now[2]+height]

        self.moveto(cam_id, location_exp)
        if angle != 0 or pitch != 0:
            self.set_rotation(cam_id, [0, yaw_exp, pitch_exp])

        location_now = self.get_location(cam_id)
        error = self.get_distance(location_now, location_exp, 2)

        if error < 10:
            return False
        else:
            return True

    def get_distance(self, pos_now, pos_exp, n=2):
        error = np.array(pos_now[:n]) - np.array(pos_exp[:n])
        distance = np.linalg.norm(error)
        return distance

    def keyboard(self, key, duration=0.01):  # Up Down Left Right
        cmd = 'vset /action/keyboard {key} {duration}'
        return self.client.request(cmd.format(key=key, duration=duration))

    def get_obj_color(self, obj):
        object_rgba = self.client.request('vget /object/{obj}/color'.format(obj=obj))
        object_rgba = re.findall(r"\d+\.?\d*", object_rgba)
        color = [int(i) for i in object_rgba]  # [r,g,b,a]
        return color[:-1]

    def set_obj_color(self, obj, color):
        cmd = 'vset /object/{obj}/color {r} {g} {b}'
        self.client.request(cmd.format(obj=obj, r=color[0], g=color[1], b=color[2]))

    def set_obj_location(self, obj, loc):
        cmd = 'vset /object/{obj}/location {x} {y} {z}'
        self.client.request(cmd.format(obj=obj, x=loc[0], y=loc[1], z=loc[2]))

    def set_obj_rotation(self, obj, rot):
        cmd = 'vset /object/{obj}/rotation {pitch} {yaw} {roll}'
        self.client.request(cmd.format(obj=obj, roll=rot[0], yaw=rot[1], pitch=rot[2]))

    def get_mask(self, object_mask, obj):
        [r, g, b] = self.color_dict[obj]
        lower_range = np.array([b-3, g-3, r-3])
        upper_range = np.array([b+3, g+3, r+3])
        mask = cv2.inRange(object_mask, lower_range, upper_range)
        return mask

    def get_bbox(self, object_mask, obj):
        # get an object's bounding box
        width = object_mask.shape[1]
        height = object_mask.shape[0]
        mask = self.get_mask(object_mask, obj)
        nparray = np.array([[[0, 0]]])
        pixelpointsCV2 = cv2.findNonZero(mask)

        if type(pixelpointsCV2) == type(nparray):  # exist target in image
            x_min = pixelpointsCV2[:, :, 0].min()
            x_max = pixelpointsCV2[:, :, 0].max()
            y_min = pixelpointsCV2[:, :, 1].min()
            y_max = pixelpointsCV2[:, :, 1].max()
            box = ((x_min/float(width), y_min/float(height)),  # left top
                   (x_max/float(width), y_max/float(height)))  # right down
        else:
            box = ((0, 0), (0, 0))

        return mask, box

    def get_bboxes(self, object_mask, objects):
        boxes = []
        for obj in objects:
            mask, box = self.get_bbox(object_mask, obj)
            boxes.append(box)
        return boxes

    def get_bboxes_obj(self, object_mask, objects):
        boxes = dict()
        for obj in objects:
            mask, box = self.get_bbox(object_mask, obj)
            boxes[obj] = box
        return boxes

    def build_color_dic(self, objects):
        color_dict = dict()
        for obj in objects:
            color = self.get_obj_color(obj)
            color_dict[obj] = color
        return color_dict

    def get_obj_location(self, obj):
        location = None
        while location is None:
            location = self.client.request('vget /object/{obj}/location'.format(obj=obj))
        return [float(i) for i in location.split()]

    def get_obj_rotation(self, obj):
        rotation = None
        while rotation is None:
            rotation = self.client.request('vget /object/{obj}/rotation'.format(obj=obj))
        return [float(i) for i in rotation.split()]

    def get_obj_pose(self, obj):
        loc = self.get_obj_location(obj)
        rot = self.get_obj_rotation(obj)
        pose = loc+rot
        return pose

    def build_pose_dic(self, objects):
        pose_dic = dict()
        for obj in objects:
            pose = self.get_obj_location(obj)
            pose.extend(self.get_obj_rotation(obj))
            pose_dic[obj] = pose
        return pose_dic

    def hide_obj(self, obj):
        self.client.request('vset /object/{obj}/hide'.format(obj=obj))

    def show_obj(self, obj):
        self.client.request('vset /object/{obj}/show'.format(obj=obj))

    def hide_objects(self, objects):
        for obj in objects:
            self.hide_obj(obj)

    def show_objects(self, objects):
        for obj in objects:
            self.show_obj(obj)

    def set_fov(self, fov, cam_id=0):
        cmd = 'vset /camera/{cam_id}/horizontal_fieldofview {FOV}'
        self.client.request(cmd.format(cam_id=cam_id, FOV=fov))
