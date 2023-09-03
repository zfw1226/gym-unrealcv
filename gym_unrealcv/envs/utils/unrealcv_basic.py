import unrealcv
import cv2
import numpy as np
import math
import time
import os
import re
from io import BytesIO
import PIL.Image
import sys

class UnrealCv(object):
    def __init__(self, port, ip, env, cam_id, resolution):

        if ip == '127.0.0.1':
            self.docker = False
        else:
            self.docker = True
        self.envdir = env
        self.ip = ip
        # build a client to connect to the env
        self.client = unrealcv.Client((ip, port))
        self.client.connect()
        if 'linux' in sys.platform and unrealcv.__version__ >= '1.0.0': # new socket for linux
            unix_socket_path = os.path.join('/tmp/unrealcv_{port}.socket'.format(port=port)) # clean the old socket
            os.remove(unix_socket_path) if os.path.exists(unix_socket_path) else None
            self.client.disconnect() # disconnect the client for creating a new socket in linux
            time.sleep(2)
            if unix_socket_path is not None and os.path.exists(unix_socket_path):
                self.client = unrealcv.Client(unix_socket_path, 'unix')
            else:
                self.client = unrealcv.Client((ip, port)) # reconnect to the tcp socket
            self.client.connect()
        self.cam = dict()
        self.color_dict = dict()
        for i in range(20):
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
        self.check_connection()
        [w, h] = resolution
        self.client.request(f'vrun setres {w}x{h}w', -1)  # set resolution of the display window
        self.client.request('DisableAllScreenMessages', -1)  # disable all screen messages
        self.client.request('vrun sg.ShadowQuality 0', -1)  # set shadow quality to low
        self.client.request('vrun sg.TextureQuality 0', -1)  # set texture quality to low
        self.client.request('vrun sg.EffectsQuality 0', -1)  # set effects quality to low
        # self.client.request('vrun r.ScreenPercentage 10')
        time.sleep(0.1)
        self.get_rotation(cam_id, 'hard')
        self.get_location(cam_id, 'hard')
        self.client.message_handler = self.message_handler

    def message_handler(self, message):
        msg = message

    def check_connection(self):
        while self.client.isconnected() is False:
            print ('UnrealCV server is not running. Please try again')
            time.sleep(1)
            self.client.connect()

    def show_img(self, img, title="raw_img"):
        cv2.imshow(title, img)
        cv2.waitKey(3)

    def get_objects(self): # get all objects in the scene
        objects = self.client.request('vget /objects').split()
        return objects

    def read_image(self, cam_id, viewmode, mode='direct'):
            # cam_id:0 1 2 ...
            # viewmode:lit,  =normal, depth, object_mask
            # mode: direct, file
            res = None
            if mode == 'direct': # get image from unrealcv in png format
                cmd = f'vget /camera/{cam_id}/{viewmode} png'
                image = self.decode_png(self.client.request(cmd))

            elif mode == 'file': # save image to file and read it
                cmd = f'vget /camera/{cam_id}/{viewmode} {viewmode}{self.ip}.png'
                if self.docker:
                    img_dirs_docker = self.client.request(cmd)
                    img_dirs = self.envdir + img_dirs_docker[7:]
                else :
                    img_dirs = self.client.request(cmd)
                image = cv2.imread(img_dirs)
            elif mode == 'fast': # get image from unrealcv in bmp format
                cmd = f'vget /camera/{cam_id}/{viewmode} bmp'
                image = self.decode_bmp(self.client.request(cmd))
            return image

    def read_depth(self, cam_id, inverse=True): # get depth from unrealcv in npy format
        cmd = f'vget /camera/{cam_id}/depth npy'
        res = self.client.request(cmd)
        depth = self.decode_depth(res)
        if inverse:
            depth = 1/depth
        # cv2.imshow('depth', depth/depth.max())
        # cv2.waitKey(10)
        return depth

    def get_img_batch(self, cam_ids, viewmode='lit', mode='bmp', inverse=True):
        # get image from multiple cameras
        # viewmode : {'lit', 'depth', 'normal', 'object_mask'}
        # mode : {'bmp', 'npy', 'png'}
        # inverse : whether to inverse the depth
        cmd_list = []
        for cam_id in cam_ids:
            cmd_list.append(f'vget /camera/{cam_id}/{viewmode} {mode}')
        res_list = self.client.request(cmd_list)
        img_list = []
        for res in res_list:
            if mode == 'npy':
                img = self.decode_depth(res)
                if inverse:
                    img = 1/img
            elif mode == 'bmp':
                img = self.decode_bmp(res)
            elif mode == 'png':
                img = self.decode_png(res)
            img_list.append(img)
        return img_list

    def decode_png(self, res): # decode png image
        img = np.asarray(PIL.Image.open(BytesIO(res)))
        img = img[:, :, :-1]  # delete alpha channel
        img = img[:, :, ::-1]  # transpose channel order
        return img

    def decode_bmp(self, res, channel=4): # decode bmp image
        img = np.fromstring(res, dtype=np.uint8)
        img=img[-self.resolution[1]*self.resolution[0]*channel:]
        img=img.reshape(self.resolution[1], self.resolution[0], channel)
        return img[:, :, :-1] # delete alpha channel

    def decode_depth(self, res):  # decode depth image
        depth = np.fromstring(res, np.float32)
        depth = depth[-self.resolution[1] * self.resolution[0]:]
        depth = depth.reshape(self.resolution[1], self.resolution[0], 1)
        return depth

    def convert2planedepth(self, PointDepth, f=320): # convert point depth to plane depth
        H = PointDepth.shape[0]
        W = PointDepth.shape[1]
        i_c = np.float(H) / 2 - 1
        j_c = np.float(W) / 2 - 1
        columns, rows = np.meshgrid(np.linspace(0, W - 1, num=W), np.linspace(0, H - 1, num=H))
        DistanceFromCenter = ((rows - i_c) ** 2 + (columns - j_c) ** 2) ** 0.5
        PlaneDepth = PointDepth / (1 + (DistanceFromCenter / f) ** 2) ** 0.5
        return PlaneDepth

    def get_rgbd(self, cam_id, mode): # get rgb and depth image
        rgb = self.read_image(cam_id, 'lit', mode)
        d = self.read_depth(cam_id)
        rgbd = np.append(rgb, d, axis=2)
        return rgbd

    def get_pose_img_batch(self, objs_list, cam_ids, img_flag=[False, True, False, False]):
        # get pose and image of objects in objs_list from cameras in cam_ids
        cmd_list = []
        [use_cam_pose, use_color, use_mask, use_depth] = img_flag
        for obj in objs_list:
            cmd_list.extend([f'vget /object/{obj}/location', f'vget /object/{obj}/rotation'])

        for cam_id in cam_ids:
            if cam_id < 0:
                continue
            if use_cam_pose:
                cmd_list.extend([f'vget /camera/{cam_id}/location', f'vget /camera/{cam_id}/rotation'])
            if use_color:
                cmd_list.append(f'vget /camera/{cam_id}/lit bmp')
            if use_mask:
                cmd_list.append(f'vget /camera/{cam_id}/object_mask bmp')
            if use_depth:
                cmd_list.append(f'vget /camera/{cam_id}/depth npy')

        res_list = self.client.request(cmd_list)
        obj_pose_list = []
        cam_pose_list = []
        img_list = []
        mask_list = []
        depth_list = []
        # start to read results
        start_point = 0
        for i, obj in enumerate(objs_list):
            loc = [float(j) for j in res_list[start_point].split()]
            start_point += 1
            rot = [float(j) for j in res_list[start_point].split()]
            start_point += 1
            obj_pose_list.append(loc + rot)
        for i, cam_id in enumerate(cam_ids):
            # print(cam_id)
            if cam_id < 0:
                img_list.append(np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8))
                continue
            if use_cam_pose:
                loc = [float(j) for j in res_list[start_point].split()]
                start_point += 1
                rot = [float(j) for j in res_list[start_point].split()][::-1]
                start_point += 1
                cam_pose_list.append(loc + rot)
            if use_color:
                image = self.decode_bmp(res_list[start_point])
                img_list.append(image)
                start_point += 1
            if use_mask:
                image = self.decode_bmp(res_list[start_point])
                mask_list.append(image)
                start_point += 1
            if use_depth:
                image = 1 / self.decode_depth(res_list[start_point])
                depth_list.append(image)  # 500 is the default max depth of most depth cameras
                start_point += 1

        return obj_pose_list, cam_pose_list, img_list, mask_list, depth_list


    def set_pose(self, cam_id, pose):  # set camera pose, pose = [x, y, z, roll, yaw, pitch]
        [x, y, z, roll, yaw, pitch] = pose
        cmd = f'vset /camera/{cam_id}/pose {x} {y} {z} {pitch} {yaw} {roll}'
        self.client.request(cmd, -1)
        self.cam[cam_id]['location'] = pose[:3]
        self.cam[cam_id]['rotation'] = pose[-3:]

    def get_pose(self, cam_id, mode='hard'):  # get camera pose, pose = [x, y, z, roll, yaw, pitch]
        if mode == 'soft':
            pose = self.cam[cam_id]['location']
            pose.extend(self.cam[cam_id]['rotation'])
            return pose
        if mode == 'hard':
            cmd = f'vget /camera/{cam_id}/pose'
            res = None
            while res is None:
                res = self.client.request(cmd)
            res = [float(i) for i in res.split()]
            self.cam[cam_id]['location'] = res[:3]
            self.cam[cam_id]['rotation'] = res[-3:]
            return res

    def set_fov(self, cam_id, fov):  # set camera field of view (fov)
        if fov == self.cam[cam_id]['fov']:
            return fov
        cmd = f'vset /camera/{cam_id}/fov {fov}'
        self.client.request(cmd, -1)
        self.cam[cam_id]['fov'] = fov
        return fov

    def get_fov(self, cam_id):  # set camera field of view (fov)
        cmd = f'vget /camera/{cam_id}/fov'
        fov = self.client.request(cmd)
        self.cam[cam_id]['fov'] = float(fov)
        return fov

    def set_location(self, cam_id, loc):  # set camera location, loc=[x,y,z]
        [x, y, z] = loc
        cmd = f'vset /camera/{cam_id}/location {x} {y} {z}'
        self.client.request(cmd, -1)
        self.cam[cam_id]['location'] = loc

    def get_location(self, cam_id, mode='hard'):
    # get camera location, loc=[x,y,z]
    # hard mode will get location from unrealcv, soft mode will get location from self.cam
        if mode == 'soft':
            return self.cam[cam_id]['location']
        if mode == 'hard':
            cmd = f'vget /camera/{cam_id}/location'
            res = None
            while res is None:
                res = self.client.request(cmd)
            self.cam[cam_id]['location'] = [float(i) for i in res.split()]
            return self.cam[cam_id]['location']

    def set_rotation(self, cam_id, rot):  # set camera rotation, rot = [roll, yaw, pitch]
        [roll, yaw, pitch] = rot
        cmd = f'vset /camera/{cam_id}/rotation {pitch} {yaw} {roll}'
        self.client.request(cmd, -1)
        self.cam[cam_id]['rotation'] = rot

    def get_rotation(self, cam_id, mode='hard'):
    # get camera rotation, rot = [roll, yaw, pitch]
    # hard mode will get rotation from unrealcv, soft mode will get rotation from self.cam
        if mode == 'soft':
            return self.cam[cam_id]['rotation']
        if mode == 'hard':
            cmd = f'vget /camera/{cam_id}/rotation'
            rotation = None
            while rotation is None:
                rotation = self.client.request(cmd)
            rotation = [float(i) for i in rotation.split()]
            rotation.reverse()
            self.cam[cam_id]['rotation'] = rotation
            return self.cam[cam_id]['rotation']

    def moveto(self, cam_id, loc): # move camera to location with physics simulation
        [x, y, z] = loc
        cmd = f'vset /camera/{cam_id}/moveto {x} {y} {z}'
        self.client.request(cmd)

    def move_2d(self, cam_id, angle, length, height=0, pitch=0):
        # move camera in 2d plane as a mobile robot
        # angle is the angle between camera and x axis
        # length is the distance between camera and target point

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

    def get_distance(self, pos_now, pos_exp, n=2): # get distance between two points, n is the dimension
        error = np.array(pos_now[:n]) - np.array(pos_exp[:n])
        distance = np.linalg.norm(error)
        return distance

    def keyboard(self, key, duration=0.01):  # Up Down Left Right
        cmd = 'vset /action/keyboard {key} {duration}'
        return self.client.request(cmd.format(key=key, duration=duration), -1)

    def get_obj_color(self, obj): # get object color in object mask, color = [r,g,b]
        cmd = f'vget /object/{obj}/color'
        object_rgba = self.client.request(cmd)
        object_rgba = re.findall(r"\d+\.?\d*", object_rgba)
        color = [int(i) for i in object_rgba]  # [r,g,b,a]
        return color[:-1]

    def set_obj_color(self, obj, color): # set object color in object mask, color = [r,g,b]
        [r, g, b] = color
        cmd = f'vset /object/{obj}/color {r} {g} {b}'
        self.client.request(cmd, -1) # -1 means async mode
        self.color_dict[obj] = color

    def set_obj_location(self, obj, loc): # set object location, loc=[x,y,z]
        [x, y, z] = loc
        cmd = f'vset /object/{obj}/location {x} {y} {z}'
        self.client.request(cmd, -1) # -1 means async mode

    def set_obj_rotation(self, obj, rot): # set object rotation, rot = [roll, yaw, pitch]
        [roll, yaw, pitch] = rot
        cmd = f'vset /object/{obj}/rotation {pitch} {yaw} {roll}'
        self.client.request(cmd, -1)

    def get_mask(self, object_mask, obj): # get an object's mask
        [r, g, b] = self.color_dict[obj]
        lower_range = np.array([b-3, g-3, r-3])
        upper_range = np.array([b+3, g+3, r+3])
        mask = cv2.inRange(object_mask, lower_range, upper_range)
        return mask

    def get_bbox(self, object_mask, obj, normalize=True): # get an object's bounding box
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
            if normalize:
                box = ((x_min/float(width), y_min/float(height)),  # left top
                       (x_max/float(width), y_max/float(height)))  # right down
            else:
                box = [x_min, y_min, x_max-x_min, y_max-y_min]
        else:
            if normalize:
                box = ((0, 0), (0, 0))
            else:
                box = [0, 0, 0, 0]

        return mask, box

    def get_bboxes(self, object_mask, objects): # get objects' bounding boxes in a image given object list, return a list
        boxes = []
        for obj in objects:
            mask, box = self.get_bbox(object_mask, obj)
            boxes.append(box)
        return boxes

    def get_bboxes_obj(self, object_mask, objects): # get objects' bounding boxes in a image given object list, return a dictionary
        boxes = dict()
        for obj in objects:
            mask, box = self.get_bbox(object_mask, obj)
            boxes[obj] = box
        return boxes

    def build_color_dic(self, objects): # build a color dictionary for objects
        color_dict = dict()
        for obj in objects:
            color = self.get_obj_color(obj)
            color_dict[obj] = color
        self.color_dict = color_dict
        return color_dict

    def get_obj_location(self, obj): # get object location
        res = None
        while res is None:
            res = self.client.request(f'vget /object/{obj}/location')
        return [float(i) for i in res.split()]

    def get_obj_rotation(self, obj): # get object rotation
        res = None
        while res is None:
            res = self.client.request(f'vget /object/{obj}/rotation')
        return [float(i) for i in res.split()]

    def get_obj_pose(self, obj): # get object pose
        loc = self.get_obj_location(obj)
        rot = self.get_obj_rotation(obj)
        pose = loc+rot
        return pose

    def build_pose_dic(self, objects): # build a pose dictionary for objects
        pose_dic = dict()
        for obj in objects:
            pose = self.get_obj_location(obj)
            pose.extend(self.get_obj_rotation(obj))
            pose_dic[obj] = pose
        return pose_dic

    def get_obj_bounds(self, obj): # get object location
        res = None
        while res is None:
            res = self.client.request(f'vget /object/{obj}/bounds')
        return [float(i) for i in res.split()] # min x,y,z  max x,y,z

    def get_obj_size(self, obj):
        # return the size of the bounding box
        self.set_obj_rotation(obj, [0, 0, 0])  # init
        bounds = self.get_obj_bounds(obj)
        x = bounds[3] - bounds[0]
        y = bounds[4] - bounds[1]
        z = bounds[5] - bounds[2]
        return [x, y, z]

    def get_obj_scale(self, obj):
        # set object scale
        res = None
        while res is None:
            res = self.client.request(f'vget /object/{obj}/scale')
        return [float(i) for i in res.split()] # [scale_x, scale_y, scale_z]

    def set_obj_scale(self, obj, scale=[1, 1, 1]):
        # set object scale
        [x, y, z] = scale
        self.client.request(f'vset /object/{obj}/scale {x} {y} {z}', -1)

    def hide_obj(self, obj): # hide an object, make it invisible, but still there in physics engine
        self.client.request(f'vset /object/{obj}/hide', -1)

    def show_obj(self, obj): # show an object, make it visible
        self.client.request(f'vset /object/{obj}/show', -1)

    def hide_objects(self, objects):
        for obj in objects:
            self.hide_obj(obj)

    def show_objects(self, objects):
        for obj in objects:
            self.show_obj(obj)

    def destroy_obj(self, obj): # destroy an object, remove it from the scene
        self.client.request(f'vset /object/{obj}/destroy', -1)

    def get_camera_num(self):
        res = self.client.request('vget /cameras')
        return len(res.split())

    def new_camera(self):
        res = self.client.request('vset /cameras/spawn')
        return res  #  return the object name of the new camera

    def get_vertex_locations(self, obj):
        res = None
        while res is None:
            res = self.client.request(f'vget /object/{obj}/vertices')
        lines = res.split('\n')
        lines = [line.strip() for line in lines]
        vertex_locations = [list(map(float, line.split())) for line in lines]
        return vertex_locations

    def set_map(self, map_name):  # change to a new level map
        self.client.request(f'vset /action/game/level {map_name}', -1)

    def set_global_time_dilation(self, time_dilation):
        self.client.request(f'vrun slomo {time_dilation}', -1)

    def set_max_FPS(self, max_fps):
        self.client.request(f'vrun t.maxFPS {max_fps}', -1)