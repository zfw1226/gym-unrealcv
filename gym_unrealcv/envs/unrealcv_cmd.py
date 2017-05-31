import unrealcv
import cv2
import numpy as np
import math
import time
import os
import re


####TO DO#######
###the observation in memory replay only save the image dir instead of feature####
###do not delet the image right away, save them and only detet it when buffer is filled totally.
class UnrealCv:
    def __init__(self,cam_id = 0, ip = '127.0.0.1' , targets = None,env = '/home/zfw/Documents/Realistic5'):
        global client
        if ip == '127.0.0.1':
            self.docker = False
        else:
            self.docker = True
        client = unrealcv.Client((ip, 9000))
        self.cam_id = cam_id
        self.envdir = env

        self.cam =dict(
                 id = cam_id,
                 position = [0,0,0],
                 x = 0,
                 y = 0,
                 z = 0,
                 roll = 0,
                 yaw  = 0,
                 pitch = 0,
        )

        self.init_unrealcv()
        self.pitch = 0 #-30
        self.color_dict = dict()
        if targets:
            self.color_dict = self.target_color_dic(targets)







    def init_unrealcv(self):
        client.connect()
        self.check_connection()
        client.request('vrun setres 320x240w')# this will set the resolution of object_mask
        self.get_position(self.cam['id'])
        self.get_rotation(self.cam['id'])
        #(x, y, z) = self.get_position(self.cam_id)
        #roll,yaw,pitch = self.get_rotation(self.cam_id)
        #return x, y, z ,yaw

    def check_connection(self):
        while (client.isconnected() is False):
            print 'UnrealCV server is not running. Please try again'
            client.connect()

    def show_img(self,img,title="raw_img"):
        cv2.imshow(title, img)
        cv2.waitKey(3)

    def get_objects(self):
        objects = client.request('vget /objects')
        return objects

    def read_image(self,cam_id , viewmode, show=False):
            # cam_id:0 1 2 ...
            # viewmode:lit,  =normal, depth, object_mask
            cmd = 'vget /camera/{cam_id}/{viewmode}'

            if self.docker:
                img_dirs_docker = client.request(cmd.format(cam_id=cam_id, viewmode=viewmode))
                img_dirs = self.envdir + img_dirs_docker[7:]
            else :
                img_dirs = client.request(cmd.format(cam_id=cam_id, viewmode=viewmode))

            image = cv2.imread(img_dirs)
            if show is True:
                self.show_img(image)
            os.remove(img_dirs)
            return image



    def set_position(self,cam_id, x, y, z):
        cmd = 'vset /camera/{cam_id}/location {x} {y} {z}'
        client.request(cmd.format(cam_id=cam_id, x=x, y=y, z=z))
        self.cam['position'] = [x,y,z]


    def set_rotation(self,cam_id, roll, yaw, pitch):
        cmd = 'vset /camera/{cam_id}/rotation {pitch} {yaw} {roll}'
        client.request(cmd.format(cam_id=cam_id, pitch=pitch, yaw=yaw, roll=roll))
        self.cam['rotation'] = (roll,yaw,pitch)

    def get_position(self,cam_id):
        cmd = 'vget /camera/{cam_id}/location'
        pos = client.request(cmd.format(cam_id=cam_id))
        x,y,z = pos.split()
        self.cam['position'] = [float(x), float(y), float(z)]
        return self.cam['position']

    def get_rotation(self,cam_id):
        cmd = 'vget /camera/{cam_id}/rotation'
        ori = client.request(cmd.format(cam_id=cam_id))
        pitch,yaw,roll = ori.split()
        self.cam['rotation'] = (float(roll), float(yaw), float(pitch))
        return self.cam['rotation']

    def moveto(self,cam_id, x, y, z):
        cmd = 'vset /camera/{cam_id}/moveto {x} {y} {z}'
        client.request(cmd.format(cam_id=cam_id, x=x, y=y, z=z))

    def move(self,cam_id, angle, length):

        yaw_exp = (self.cam['rotation'][1] + angle) % 360
        delt_x = length * math.cos(yaw_exp / 180.0 * math.pi)
        delt_y = length * math.sin(yaw_exp / 180.0 * math.pi)
        x_exp = self.cam['position'][0] + delt_x
        y_exp = self.cam['position'][1] + delt_y
        z_exp = self.cam['position'][2]

        self.moveto(cam_id, x_exp, y_exp, z_exp)

        if angle != 0 :
            self.set_rotation(cam_id, 0, yaw_exp, self.pitch)

        self.cam['position'] = self.get_position(cam_id)
        error = self.error_position(self.cam['position'],(x_exp,y_exp,z_exp))

        if (error < 10):
            return False
        else:
            return True


    def error_position(self,pos_now,pos_exp):
        pos_error = (np.array(pos_now) - np.array(pos_exp)) ** 2
        return pos_error.mean()


    def keyboard(self,key, duration = 0.3):# Up Down Left Right
        cmd = 'vset /action/keyboard {key} {duration}'
        return client.request(cmd.format(key = key,duration = duration))

    def get_object_color(self,object):
        object_rgba = client.request('vget /object/' + object + '/color')
        object_rgba = re.findall(r"\d+\.?\d*", object_rgba)
        r = int(object_rgba[0])
        g = int(object_rgba[1])
        b = int(object_rgba[2])
        return r,g,b


    def set_object_color(self,object,r,g,b):
        setcolor = 'vset /object/'+object+'/color {r} {g} {b}'
        client.request(setcolor.format(r=r, g=g, b=b))

    def get_mask(self,object_mask,object):
        r,g,b = self.color_dict[object]
        lower_range = np.array([b-5,g-5,r-5])
        upper_range = np.array([b+5,g+5,r+5])
        mask = cv2.inRange(object_mask, lower_range, upper_range)
        return mask

    def get_mask_binary(self,object_mask):
        gray = cv2.cvtColor(object_mask,cv2.COLOR_RGB2GRAY)
        ret, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        print ret
        return mask

    def get_bbox(self,object_mask,object):
        #only get an object's bounding box
        width = object_mask.shape[1]
        height = object_mask.shape[0]
        mask = self.get_mask(object_mask,object)
        nparray = np.array([[[0, 0]]])
        pixelpointsCV2 = cv2.findNonZero(mask)

        if type(pixelpointsCV2) == type(nparray):# exist target in image
            x_min = pixelpointsCV2[:,:,0].min()
            x_max = pixelpointsCV2[:,:,0].max()
            y_min = pixelpointsCV2[:,:,1].min()
            y_max = pixelpointsCV2[:,:,1].max()
            #print x_min, x_max ,y_min, y_max
            box = ((x_min/float(width),y_min/float(height)),#left top
                   (x_max/float(width),y_max/float(height)))#right down
        else:
            box = ((0,0),(0,0))

        return mask , box

    def get_bboxes(self,object_mask,objects):
        boxes = []
        for obj in objects:
            mask,box = self.get_bbox(object_mask, obj)
            boxes.append(box)
        return  boxes

    def target_color_dic(self,objects):
        object_dict = dict()
        for obj in objects:
            color = self.get_object_color(obj)
            object_dict[obj] = color
        return object_dict

    def get_object_pos(self,object):
        pos = client.request('vget /object/{obj}/location'.format(obj = object))
        x,y,z = pos.split()
        return (float(x),float(y),float(z))

    def get_objects_pos(self,objects):
        pos = []
        for obj in objects:
            pos.append(self.get_object_pos(obj))
        return pos

    def get_pos(self):
        return self.cam['position']

    def get_pose(self):
        pos = self.cam['position'][:]
        pos.append(self.cam['rotation'][1])
        return pos

    def get_height(self):
        return self.cam['position'][2]















