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

    def get_startpoint(self, target_pos, distance, reset_area, exp_height=200, direction=None):
        count = 0
        while True:  # searching a safe point
            if direction == None:
                direction = 2 * np.pi * np.random.sample(1)
            else:
                direction = direction % (2 * np.pi)
            dx = float(distance * np.cos(direction))
            dy = float(distance * np.sin(direction))
            x = dx + target_pos[0]
            y = dy + target_pos[1]
            cam_pos_exp = [x, y, exp_height]
            yaw = float(direction / np.pi * 180 - 180)
            if reset_area[0] < x < reset_area[1] and reset_area[2] < y < reset_area[3]:
                cam_pos_exp[0] = dx + target_pos[0]
                cam_pos_exp[1] = dy + target_pos[1]
                return [cam_pos_exp, yaw]
            else:
                count += 1
                if count > 5:
                    return False

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
                self.set_skylight(lit, [1, 1, 1], np.random.uniform(0.5, 2))
            else:
                lit_direction = np.random.uniform(-1, 1, 3)
                if 'directional' in lit:
                    lit_direction[0] = lit_direction[0] * 60
                    lit_direction[1] = lit_direction[1] * 80
                    lit_direction[2] = lit_direction[2] * 60
                else:
                    lit_direction *= 180
                self.set_light(lit, lit_direction, np.random.uniform(1, 4), np.random.uniform(0.1, 1, 3))

    def set_random(self, target, value=1):
        cmd = 'vbp {target} set_random {value}'.format(target=target, value=value)
        res=None
        while res is None:
            res = self.client.request(cmd)

    def set_interval(self, interval):
        cmd = 'vbp set_interval {value}'.format(value=interval)
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

    def random_obstacles(self, objects, img_dirs, num, area, start_area, obstacle_scales=None):
        sample_index = np.random.choice(len(objects), num, replace=False)
        for k, id in enumerate(sample_index):
            obstacle = objects[id]
            self.obstacles.append(obstacle)
            # texture
            img_dir = img_dirs[np.random.randint(0, len(img_dirs))]
            self.set_texture(obstacle, (1, 1, 1), np.random.uniform(0, 1, 3), img_dir, np.random.randint(1, 4))
            # scale
            if obstacle_scales == None:
                obs_scale = [0.3, 3]
            else:
                obs_scale = obstacle_scales[k]
            self.set_obj_scale(obstacle, np.random.uniform(obs_scale[0], obs_scale[1], 3))
            # location
            obstacle_loc = [start_area[0], start_area[2], 0]
            while start_area[0] <= obstacle_loc[0] <= start_area[1] and start_area[2] <= obstacle_loc[1] <= start_area[3]:
                obstacle_loc[0] = np.random.uniform(area[0]+100, area[1]-100)
                obstacle_loc[1] = np.random.uniform(area[2]+100, area[3]-100)
                obstacle_loc[2] = np.random.uniform(area[4], area[5])
            self.set_obj_location(obstacle, obstacle_loc)
            time.sleep(0.03)

    def clean_obstacles(self):
        for obj in self.obstacles:
            self.set_obj_location(obj, self.objects_dict[obj])
        self.obstacles = []

    def new_obj(self, obj_type, loc, rot=[0, 0, 0]):
        # return obj name
        cmd = 'vbp spawn spawn {x} {y} {z} {roll} {pitch} {yaw} {obj_type}'.format(
            obj_type=obj_type, x=loc[0], y=loc[1], z=loc[2], roll=rot[0], pitch=rot[1], yaw=rot[2])
        res = None
        while res is None:
            res = self.client.request(cmd)
        return res[12:-3]

    def destroy_obj(self, obj):
        # return obj name
        cmd = 'vbp {obj} destroy'.format(obj=obj)
        res = None
        while res is None:
            res = self.client.request(cmd)
        return res[12:-3]

    def move_goal(self, obj, goal):
        cmd = 'vbp {obj} move_to_goal {x} {y}'.format(obj=obj, x=goal[0] , y=goal[1])
        res = None
        while res is None:
            res = self.client.request(cmd)

