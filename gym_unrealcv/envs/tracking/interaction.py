from gym_unrealcv.envs.navigation.interaction import Navigation
import numpy as np
import math
import time


class Tracking(Navigation):
    def __init__(self, env, cam_id=0, port=9000,
                 ip='127.0.0.1', resolution=(160, 120)):
        super(Tracking, self).__init__(env=env, port=port, ip=ip, cam_id=cam_id, resolution=resolution)
        self.obstacles = []

    def random_texture(self, backgrounds, img_dirs, num=5):
        if num < 0:
            sample_index = range(len(backgrounds))
        else:
            sample_index = np.random.choice(len(backgrounds), num, replace=False)
        for id in sample_index:
            target = backgrounds[id]
            img_dir = img_dirs[np.random.randint(0, len(img_dirs))]
            self.set_texture(target, (1, 1, 1), np.random.uniform(0, 1, 3), img_dir, np.random.randint(1, 4))
            time.sleep(0.03)

    def random_player_texture(self, player, img_dirs, num):
        sample_index = np.random.choice(5, num)
        for id in sample_index:
            img_dir = img_dirs[np.random.randint(0, len(img_dirs))]
            self.set_texture(player, (1, 1, 1), np.random.uniform(0, 1, 3),
                             img_dir, np.random.randint(2, 6), id)
            time.sleep(0.03)

    def random_character(self, target):  # appearance, speed, acceleration
        self.set_speed(target, np.random.randint(40, 100))
        self.set_acceleration(target, np.random.randint(100, 300))
        self.set_maxdis2goal(target, np.random.randint(200, 3000))

    # functions for character setting
    def set_speed(self, target, speed):
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

    def set_maxdis2goal(self, target, dis):
        cmd = 'vbp {target} set_maxrange {dis}'
        res = None
        while res is None:
            res = self.client.request(cmd.format(target=target, dis=dis))
        return dis

    def set_appearance(self, target, id, spline=False):
        if spline:
            cmd = 'vbp {target} set_app {id}'
        else:
            cmd = 'vbp {target} set_mixamoapp {id}'
        res = None
        while res is None:
            res = self.client.request(cmd.format(target=target, id=id))
        return id

    def start_walking(self, target):
        cmd = 'vbp {target} start'
        res = None
        while res is None:
            res = self.client.request(cmd.format(target=target))
        if 'true' in res:
            return True
        if 'false' in res:
            return False

    def get_pose(self, cam_id, mode='hard'):  # pose = [x, y, z, roll, yaw, pitch]
        if mode == 'soft':
            pose = self.cam[cam_id]['location']
            pose.extend(self.cam[cam_id]['rotation'])
            return pose

        if mode == 'hard':
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

        self.moveto(cam_id, location_exp)
        if angle != 0:
            self.set_rotation(cam_id, [0, yaw_exp, self.pitch])

        location_now = self.get_location(cam_id)
        error = self.get_distance(location_now, location_exp)

        if error < 10:
            return False
        else:
            return True

    def get_startpoint(self, target_pos=[], distance=None, reset_area=[], exp_height=200, direction=None):
        for i in range(5):  # searching a safe point
            if direction == None:
                direction = 2 * np.pi * np.random.sample(1)
            else:
                direction = direction % (2 * np.pi)
            if distance == None:
                x = np.random.randint(reset_area[0], reset_area[1])
                y = np.random.randint(reset_area[2], reset_area[3])
            else:
                dx = float(distance * np.cos(direction))
                dy = float(distance * np.sin(direction))
                x = dx + target_pos[0]
                y = dy + target_pos[1]
            cam_pos_exp = [x, y, exp_height]
            yaw = float(direction / np.pi * 180 - 180)
            if reset_area[0] < x < reset_area[1] and reset_area[2] < y < reset_area[3]:
                cam_pos_exp[0] = x
                cam_pos_exp[1] = y
                return [cam_pos_exp, yaw]
        return []

    def reset_target(self, target):
        cmd = 'vbp {target} reset'
        res=None
        while res is None:
            res = self.client.request(cmd.format(target=target))

    def set_phy(self, obj, state):
        cmd = 'vbp {target} set_phy {state}'
        res=None
        while res is None:
            res = self.client.request(cmd.format(target=obj, state =state))

    def simulate_physics(self, objects):
        for obj in objects:
            self.set_phy(obj, 1)

    def set_move(self, target, angle, velocity):
        cmd = 'vbp {target} set_move {angle} {velocity}'.format(target=target, angle=angle, velocity=velocity)
        res = None
        while res is None:
            res = self.client.request(cmd)

    def set_move_batch(self, objs_list, action_list):
        cmd = 'vbp {obj} set_move {angle} {velocity}'
        cmd_list = []
        for i in range(len(objs_list)):
            cmd_list.append(cmd.format(obj=objs_list[i], angle=action_list[i][1], velocity=action_list[i][0]))
        res = self.client.request(cmd_list, -1) # -1 means async request

    def set_move_with_cam_batch(self, objs_list, action_list, cam_ids, cam_rots):
        cmd_move = 'vbp {obj} set_move {angle} {velocity}'
        cmd_rot_cam = 'vset /camera/{cam_id}/rotation {pitch} {yaw} {roll}'
        cmd_list = []
        for i in range(len(objs_list)):
            cmd_list.append(cmd_move.format(obj=objs_list[i], angle=action_list[i][1], velocity=action_list[i][0]))
        for i in range(len(cam_ids)):
            rot = cam_rots[i]
            cam_id = cam_ids[i]
            self.client.request(cmd_rot_cam.format(cam_id=cam_id, roll=rot[0], yaw=rot[1], pitch=rot[2]))
            self.cam[cam_id]['rotation'] = rot
        res = self.client.request(cmd_list, -1) # -1 means async request

    def get_hit(self, target):
        cmd = 'vbp {target} get_hit'.format(target=target)
        res = None
        while res is None:
            res = self.client.request(cmd)
        if 'true' in res:
            return True
        if 'false' in res:
            return False

    def random_lit(self, light_list):
        for lit in light_list:
            if 'sky' in lit:
                self.set_skylight(lit, [1, 1, 1], np.random.uniform(1, 10))
            else:
                lit_direction = np.random.uniform(-1, 1, 3)
                if 'directional' in lit:
                    lit_direction[0] = lit_direction[0] * 60
                    lit_direction[1] = lit_direction[1] * 80
                    lit_direction[2] = lit_direction[2] * 60
                else:
                    lit_direction *= 180
                self.set_light(lit, lit_direction, np.random.uniform(1, 5), np.random.uniform(0.3, 1, 3))

    def set_random(self, target, value=1):
        cmd = 'vbp {target} set_random {value}'.format(target=target, value=value)
        res=None
        while res is None:
            res = self.client.request(cmd)

    def set_interval(self, interval, target=None):
        if target is None:
            cmd = 'vbp set_interval {value}'.format(value=interval)
        else:
            cmd = 'vbp {target} set_interval {value}'.format(target=target, value=interval)
        res = None
        while res is None:
            res = self.client.request(cmd)

    def init_objects(self, objects):
        self.objects_dict = dict()
        for obj in objects:
            print (obj)
            self.objects_dict[obj] = self.get_obj_location(obj)
        return self.objects_dict

    def set_obj_scale(self, obj, scale):
        cmd = 'vbp {obj} set_scale {x} {y} {z}'.format(obj=obj, x=scale[0], y=scale[1], z=scale[2])
        res = None
        while res is None:
            res = self.client.request(cmd)

    def random_obstacles(self, objects, img_dirs, num, area, start_area, texture=False):
        sample_index = np.random.choice(len(objects), num, replace=False)
        for id in sample_index:
            obstacle = objects[id]
            self.obstacles.append(obstacle)
            # texture
            if texture:
                img_dir = img_dirs[np.random.randint(0, len(img_dirs))]
                self.set_texture(obstacle, (1, 1, 1), np.random.uniform(0, 1, 3), img_dir, np.random.randint(1, 4))
            # scale
            self.set_obj_scale(obstacle, np.random.uniform(0.3, 3, 3))
            # location
            obstacle_loc = [start_area[0], start_area[2], 0]
            while start_area[0] <= obstacle_loc[0] <= start_area[1] and start_area[2] <= obstacle_loc[1] <= start_area[3]:
                obstacle_loc[0] = np.random.uniform(area[0]+100, area[1]-100)
                obstacle_loc[1] = np.random.uniform(area[2]+100, area[3]-100)
                obstacle_loc[2] = np.random.uniform(area[4], area[5])
            self.set_obj_location(obstacle, obstacle_loc)
            time.sleep(0.01)

    def clean_obstacles(self):
        for obj in self.obstacles:
            self.set_obj_location(obj, self.objects_dict[obj])
        self.obstacles = []

    def new_obj(self, obj_type, obj_name, loc, rot=[0, 0, 0]):
        # spawn, set obj pose, enable physics
        cmd = ['vset /objects/spawn {0} {1}'.format(obj_type, obj_name),
               'vset /object/{0}/location {1} {2} {3}'.format(obj_name, loc[0], loc[1], loc[2]),
               'vset /object/{0}/rotation {1} {2} {3}'.format(obj_name, rot[0], rot[1], rot[2]),
               'vbp {0} set_phy 1'.format(obj_name)
               ]
        res = self.client.request(cmd, -1)
        return obj_name

    def move_goal(self, obj, goal):
        cmd = 'vbp {obj} move_to_goal {x} {y}'.format(obj=obj, x=goal[0] , y=goal[1])
        res = None
        while res is None:
            res = self.client.request(cmd)

    def get_pose_img_batch(self, objs_list, cam_ids, obs_type='lit', mode='fast', cam_rot=False):
        # get pose and image of objects in objs_list from cameras in cam_ids
        cmd_img = 'vget /camera/{cam_id}/{viewmode} bmp'
        cmd_depth = 'vget /camera/{cam_id}/depth npy'
        cmd_loc = 'vget /object/{obj}/location'
        cmd_rot = 'vget /object/{obj}/rotation'
        cmd_cam_rot = 'vget /camera/{cam_id}/rotation'
        cmd_cam_loc = 'vget /camera/{cam_id}/location'
        cmd_list = []
        if obs_type == 'Color':
            viewmode = 'lit'
        elif 'Mask' in obs_type:
            viewmode = 'object_mask'
        if 'Depth' in obs_type:
            use_depth = True
        else:
            use_depth = False

        for obj in objs_list:
            cmd_list.extend([cmd_loc.format(obj=obj), cmd_rot.format(obj=obj)])
        for cam_id in cam_ids:
            cmd_list.append(cmd_img.format(cam_id=cam_id, viewmode=viewmode, mode=mode))
            if use_depth:
                cmd_list.append(cmd_depth.format(cam_id=cam_id))
            if cam_rot:
                cmd_list.extend([cmd_cam_loc.format(cam_id=cam_id), cmd_cam_rot.format(cam_id=cam_id)])
        res_list = self.client.request(cmd_list)
        pose_list = []
        img_list = []
        depth_list = []
        for i, obj in enumerate(objs_list):
            loc = [float(j) for j in res_list[i*2].split()]
            rot = [float(j) for j in res_list[i*2+1].split()]
            pose = loc + rot
            pose_list.append(pose)
        start_point = len(objs_list)*2
        for i, cam_id in enumerate(cam_ids):
            p = int(start_point+i*(1+use_depth))
            image = self.decode_bmp(res_list[p])[:, :, :-1]
            img_list.append(image)
            if use_depth:
                depth = np.fromstring(res_list[p+1], np.float32)
                depth = depth[-self.resolution[1] * self.resolution[0]:].reshape(self.resolution[1], self.resolution[0], 1)
                depth_list.append(200/depth)
        if cam_rot:
            cam_pose_list = []
            for i, cam_id in enumerate(cam_ids):
                loc = [float(j) for j in res_list[start_point+i*2].split()]
                rot = [float(j) for j in res_list[start_point+i*2+1].split()][::-1]
                pose = loc + rot
                cam_pose_list.append(pose)
            return img_list, pose_list, cam_pose_list, depth_list
        else:
            return img_list, pose_list, depth_list

    def get_depth_batch(self, cam_ids, inverse=True):
        # get depth image from multiple cameras
        cmd_list = []
        cmd_depth = 'vget /camera/{cam_id}/depth npy'
        for cam_id in cam_ids:
            cmd_list.append(cmd_depth.format(cam_id=cam_id))
        res_list = None
        while res_list is None:
            res_list = self.client.request(cmd_list)
        depth_list = []
        for res in res_list:
            depth = np.fromstring(res, np.float32)
            depth = depth[-self.resolution[1] * self.resolution[0]:]
            depth = depth.reshape(self.resolution[1], self.resolution[0], 1)
            if inverse:
                depth = 1/depth
            depth_list.append(depth)
        return depth_list

    def get_img_batch(self, cam_ids, obs_type='lit', mode='fast'):
        # get image from multiple cameras
        cmd_list = []
        cmd_img = 'vget /camera/{cam_id}/{viewmode} bmp'
        for cam_id in cam_ids:
            cmd_list.append(cmd_img.format(cam_id=cam_id, viewmode=obs_type, mode=mode))
        res_list = None
        while res_list is None:
            res_list = self.client.request(cmd_list)
        img_list = []
        for res in res_list:
            img = self.decode_bmp(res)[:, :, :-1]
            img_list.append(img)
        return img_list

    def set_cam(self, obj, loc=[0, 30, 70], rot=[0, 0, 0]):
        # set the camera pose relative to a actor
        x, y, z = loc
        row, pitch, yaw = rot
        cmd = 'vbp {0} set_cam {1} {2} {3} {4} {5} {6}'.format(obj, x, y, z, row, pitch, yaw)
        res = self.client.request(cmd, -1)
        return res

    def adjust_fov(self, cam_id, delta_fov, min_max=[45, 135]):  # increase/decrease fov
        return self.set_fov(cam_id, np.clip(self.cam[cam_id]['fov']+delta_fov, min_max[0], min_max[1]))
