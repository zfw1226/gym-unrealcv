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
    def __init__(self, port = 9000, cam_id = 0,
                 ip = '127.0.0.1' , targets = None,
                 env = '/home/zfw/Documents/Realistic5'):
        global client
        if ip == '127.0.0.1':
            self.docker = False
        else:
            self.docker = True
        client = unrealcv.Client((ip, port))
        self.cam_id = cam_id
        self.envdir = env
        self.ip = ip

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

        self.arm = dict(
                pose = np.zeros(5),
                flag_pose = False,
                grip = np.zeros(3),
                flag_grip = False

        )

        self.init_unrealcv()
        self.pitch = 0 #-30
        self.color_dict = dict()
        self.targets = []
        if targets == 'all':
            self.targets = self.get_objects()
            self.color_dict = self.target_color_dic(self.targets)
        elif targets is not None:
            self.targets = targets
            self.color_dict = self.target_color_dic(self.targets)

        self.message = []


    def init_unrealcv(self):
        client.connect()
        self.check_connection()
        client.request('vrun setres 320x240w')# this will set the resolution of object_mask
        time.sleep(5)
        self.get_position(self.cam['id'])
        self.get_rotation(self.cam['id'])
        client.message_handler = self.message_handler

    def message_handler(self,message):

        msg = message
        #filter for pose
        if 'Currentpose' in msg:
            pose_str = msg[12:].split()
            self.arm['pose'] = np.array(pose_str,dtype=np.float16)
            self.arm['flag_pose'] = True
            #print 'get arm pose:{}'.format(self.arm['pose'])
        elif 'GripLocation' in msg:
            pose_str = msg[13:].split()
            self.arm['grip'] = np.array(pose_str, dtype=np.float16)
            self.arm['flag_grip'] = True
        elif message != 'move':
            self.message.append(message)

    def read_message(self):
        msg = self.message
        self.message = []
        return msg

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


    def check_connection(self):
        while (client.isconnected() is False):
            print 'UnrealCV server is not running. Please try again'
            client.connect()

    def show_img(self,img,title="raw_img"):
        cv2.imshow(title, img)
        cv2.waitKey(3)

    def get_objects(self):
        objects = client.request('vget /objects')
        objects = objects.split()
        return objects

    def read_image(self,cam_id , viewmode, show=False):
            # cam_id:0 1 2 ...
            # viewmode:lit,  =normal, depth, object_mask
            cmd = 'vget /camera/{cam_id}/{viewmode} {viewmode}{ip}.png'
            if self.docker:
                img_dirs_docker = client.request(cmd.format(cam_id=cam_id, viewmode=viewmode,ip=self.ip))
                img_dirs = self.envdir + img_dirs_docker[7:]
            else :
                img_dirs = client.request(cmd.format(cam_id=cam_id, viewmode=viewmode,ip=self.ip))
            image = cv2.imread(img_dirs)

            return image

    def read_depth(self, cam_id):
        cmd = 'vget /camera/{cam_id}/depth npy'
        res = client.request(cmd.format(cam_id=cam_id))
        import StringIO
        depth = np.load(StringIO.StringIO(res))
        depth[depth>100.0] = 0
        #self.show_img(depth,'depth')
        #return depth
        return np.expand_dims(depth,axis=-1)

    def convert2planedepth(self,PointDepth, f=320):
        H = PointDepth.shape[0]
        W = PointDepth.shape[1]
        i_c = np.float(H) / 2 - 1
        j_c = np.float(W) / 2 - 1
        columns, rows = np.meshgrid(np.linspace(0, W - 1, num=W), np.linspace(0, H - 1, num=H))
        DistanceFromCenter = ((rows - i_c) ** 2 + (columns - j_c) ** 2) ** (0.5)
        PlaneDepth = PointDepth / (1 + (DistanceFromCenter / f) ** 2) ** (0.5)
        return PlaneDepth

    def get_rgbd(self,cam_id):
        rgb = self.read_image(cam_id,'lit')
        d = self.read_depth(cam_id)
        rgbd = np.append(rgb,d,axis=2)
        return rgbd

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
        self.cam['rotation'] = [float(roll), float(yaw), float(pitch)]
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


    def keyboard(self,key, duration = 0.01):# Up Down Left Right
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
        lower_range = np.array([b-3,g-3,r-3])
        upper_range = np.array([b+3,g+3,r+3])
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

    def get_bboxes_obj(self,object_mask,objects):
        boxes = dict()
        for obj in objects:
            mask,box = self.get_bbox(object_mask, obj)
            boxes[obj] = box
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















