from gym_unrealcv.envs.navigation.interaction import Navigation
import numpy as np
import cv2
import math
class Tracking(Navigation):
    def __init__(self, env, cam_id = 0, port = 9000,
                 ip = '127.0.0.1', targets = None, resolution=(160,120)):
        super(Tracking, self).__init__(env=env, port = port,ip = ip , cam_id=cam_id,resolution=resolution)

    # functions for character setting
    def random_env(self,backgrounds, img_dirs, type, lights):
        for target in backgrounds:
            if np.random.sample(1) > 0.5:
                if type=='img':
                    img_dir = img_dirs[np.random.randint(0, len(img_dirs))]
                    self.set_picture(target,img_dir)
                elif type=='color':
                    self.set_color(target,np.round(np.random.sample(6),3))

        for lit in lights:
            if np.random.sample(1) > 0.5:
                self.set_light(lit, 360*np.random.sample(3), np.random.sample(1), np.random.sample(3))

    def random_character(self, target, num): #apperance, speed, acceleration
        self.set_speed(target, np.random.randint(60,160))
        self.set_acceleration(target, np.random.randint(100,500))
        self.set_maxdis2goal(target, np.random.randint(1000,3000))
        self.set_appearance(target, np.random.randint(0, num))

    def random_texture(self, backgrounds, img_dirs):
        sample_index = np.random.choice(len(backgrounds), 5)
        for id in sample_index:
            target = backgrounds[id]
            img_dir = img_dirs[np.random.randint(0, len(img_dirs))]
            self.set_texture(target,(1,1,1),np.random.uniform(0,1,3),img_dir, np.random.randint(1,4))

        self.set_texture('floor', (1,1,1), np.random.uniform(0, 1, 3), img_dirs[np.random.randint(0, len(img_dirs))],
                         np.random.randint(1, 4))

    def set_picture(self,target,dir):
        cmd = 'vbp {target} set_pic {dir}'
        res = self.client.request(cmd.format(target=target, dir=dir))
        #print (cmd.format(target=target, dir=dir))
        return res

    def set_color(self, target, param):
        cmd = 'vbp {target} set_color {r} {g} {b} {meta} {spec}'
        #cmd = 'vbp {target} setcolor {r} {g} {b} {meta} {spec} {rough}'
        self.client.request(cmd.format(target=target, r=param[0], g=param[1], b=param[2], meta=param[3], spec=param[4], rough=param[5]))
        return param


    # functions for character setting
    def set_speed(self,target, speed):
        cmd = 'vbp {target} set_speed {speed}'
        res = None
        while res is None:
            res = self.client.request(cmd.format(target=target, speed=speed))
        return speed

    def set_acceleration(self, target, acc):
        cmd = 'vbp {target} set_acc {acc}'
        res = None
        while res is None:
            res = self.client.request(cmd.format(target=target, acc=acc))
        return acc

    def set_appearance(self, target, id):
        #cmd = 'vbp {target} set_app {id}'
        cmd = 'vbp {target} set_mixamoapp {id}'
        res = None
        while res is None:
            res = self.client.request(cmd.format(target=target, id=id))
        return id

    def set_maxdis2goal(self, target, dis):
        cmd = 'vbp {target} set_maxrange {dis}'
        res = None
        while res is None:
            res = self.client.request(cmd.format(target=target, dis=dis))
        return dis

    def start_walking(self, target):
        cmd = 'vbp {target} start'
        res = None
        while res is None:
            res = self.client.request(cmd.format(target=target))
        if 'true' in res:
            return True
        if 'false' in res:
            return False



    def get_pose(self,cam_id, type='hard'):# pose = [x, y, z, roll, yaw, pitch]
        if type == 'soft':
            pose = self.cam[cam_id]['location']
            pose.extend(self.cam[cam_id]['rotation'])
            return pose

        if type == 'hard':
            self.cam[cam_id]['location'] = self.get_location(cam_id)
            self.cam[cam_id]['rotation'] = self.get_rotation(cam_id)
            pose = self.cam[cam_id]['location'] + self.cam[cam_id]['rotation']
            return pose

    def move_2d(self, cam_id, angle, length):

        yaw_exp = (self.cam[cam_id]['rotation'][1] + angle) % 360
        delt_x = length * math.cos(yaw_exp / 180.0 * math.pi)
        delt_y = length * math.sin(yaw_exp / 180.0 * math.pi)

        location_now = self.cam[cam_id]['location']
        location_exp = [location_now[0] + delt_x, location_now[1]+delt_y,location_now[2]]

        #self.set_location(cam_id, location_exp)
        self.moveto(cam_id, location_exp)
        if angle != 0 :
            self.set_rotation(cam_id, [0, yaw_exp, self.pitch])

        location_now = self.get_location(cam_id)
        error = self.error_position(location_now, location_exp)

        if (error < 10):
            return False
        else:
            return True

    def get_location_new(self,cam_id,type='hard'):
        if type == 'soft':
            return self.cam[cam_id]['location']
        if type == 'hard':
            cmd = 'vget /sensor/{cam_id}/location'
            location = None
            while location is None:
                location = self.client.request(cmd.format(cam_id=cam_id))
            self.cam[cam_id]['location'] = [float(i) for i in location.split()]
            return self.cam[cam_id]['location']

    def get_startpoint(self,target_pos, distance, reset_area):
        count = 0
        while True: # searching a safe point
            direction = 2 * np.pi * np.random.sample(1)
            dx = float(distance * np.cos(direction))
            dy = float(distance * np.sin(direction))
            cam_pos_exp = target_pos
            x = dx + target_pos[0]
            y = dy + target_pos[1]
            cam_pos_exp[2] = 150
            yaw = float(direction / np.pi * 180 - 180)
            if reset_area[0]<x<reset_area[1]  and  reset_area[2]<y<reset_area[3]:
                cam_pos_exp[0] = dx + target_pos[0]
                cam_pos_exp[1] = dy + target_pos[1]
                return [cam_pos_exp, yaw]
            else:
                count += 1
                if count > 5:
                    return False


    def reset_target(self,target):
        cmd = 'vbp {target} reset'
        res=None
        while res is None:
            res = self.client.request(cmd.format(target=target))

