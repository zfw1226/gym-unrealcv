import random
from operator import itemgetter
import math
import numpy as np
class ResetPoint():
    def __init__(self, setting, type, init_pose):
        self.reset_type = type
        #self.testpoints = setting['test_xy']
        self.waypoints = []
        self.collisionpoints = []
        self.start_id = 0
        self.yaw_id = 0
        self.waypoint_th = setting['waypoint_th']
        self.collision_th = setting['collision_th']
        self.height = setting['height']
        self.pitch = setting['pitch']
        if self.reset_type == 'testpoint':
            for x,y in setting['test_xy']:
                pose = [x,y,setting['height'],0]
                self.new_waypoint(pose, 1000)
        elif self.reset_type == 'waypoint':
            self.new_waypoint(init_pose, 1000)
        elif self.reset_type == 'random':
            self.reset_area = setting['reset_area']

    def select_resetpoint(self):
        if  'random' in self.reset_type:
            current_pose = self.reset_random()
        elif 'testpoint' in self.reset_type:
            current_pose = self.reset_testpoint()
        elif 'waypoint' in self.reset_type:
            current_pose = self.reset_waypoint()
        return current_pose

    def reset_random(self):
        x = random.uniform(self.reset_area[0], self.reset_area[1])
        y = random.uniform(self.reset_area[2], self.reset_area[3])
        z = random.uniform(self.reset_area[4], self.reset_area[5])
        yaw = random.randint(0, 360)

        return [x,y,z, 0, yaw, self.pitch]

    def reset_testpoint(self):
        x, y, z, yaw = self.waypoints[self.start_id]['pose']
        yaw = self.yaw_id * 45
        self.yaw_id += 1
        if self.yaw_id >= 8:
            self.start_id = (self.start_id + 1) % len(self.waypoints)
            self.yaw_id = self.yaw_id % 8
        return [x, y, z, 0, yaw, self.pitch]

    def reset_waypoint(self):
        # reset from waypoints generated in exploration
        x, y, z, pitch, yaw, roll = self.select_waypoint_times()
        #x, y, z, pitch, yaw, roll = self.select_waypoint_random()
        yaw = random.randint(0, 360)
        return [x, y, z, roll, yaw, self.pitch]

    def select_waypoint_times(self):

        self.waypoints = sorted(self.waypoints, key=itemgetter('selected'))
        self.start_id = random.randint(0, (len(self.waypoints) - 1) / 3)
        self.waypoints[self.start_id]['selected'] += 1

        return self.waypoints[self.start_id]['pose']

    def new_waypoint(self, pose, dis2collision):
        waypoint = dict()
        waypoint['pose'] = pose
        waypoint['successed'] = 0
        waypoint['selected'] = 0
        waypoint['dis2collision'] = dis2collision
        waypoint['steps2target'] = []
        self.waypoints.append(waypoint)
        return waypoint

    def get_dis2collision(self, pose):
        dis2collision = 1000
        for C in self.collisionpoints:
            dis2collision = min(dis2collision, self.get_distance(pose, C))
        return dis2collision


    def get_distance(self, target, current):

        error = abs(np.array(target)[:2] - np.array(current)[:2])  # only x and y
        distance = math.sqrt(sum(error * error))
        return distance

    def update_waypoint(self, trajectory):

        for P in trajectory:
            dis2waypoint, waypoint_id, dis2others = self.get_dis2waypoints(P[:3]) # searching for the closed waypoint
            dis2collision = self.get_dis2collision(P[:3])

            # update waypoint
            if (dis2waypoint < self.waypoint_th / 4 and
                dis2collision > self.waypoints[waypoint_id]['dis2collision'] and
                dis2others > self.waypoint_th):
                self.waypoints[waypoint_id]['pose'] = P
                self.waypoints[waypoint_id]['dis2collision'] = dis2collision

            # if the point is far from other existing waypoints and collision points, insert it to the waypoints list
            # add a new waypoint
            if dis2waypoint > self.waypoint_th and dis2collision > self.collision_th:
                self.new_waypoint(P, dis2collision)


        return len(self.waypoints)

    def success_waypoint(self, steps2target):
        self.waypoints[self.start_id]['successed'] += 1
        self.waypoints[self.start_id]['steps2target'].append(steps2target)

    def get_dis2waypoints(self, pose):
        dis2waypoints = []
        for W in self.waypoints:
            dis2waypoints.append(self.get_distance(pose, W['pose']))
        dis2waypoints = np.array(dis2waypoints)
        arg = np.argsort(dis2waypoints)

        id_min = arg[0]
        dis_min = dis2waypoints[id_min]
        if len(dis2waypoints) > 1:
            dis_other = dis2waypoints[arg[1]]
        else:
            dis_other = dis_min
        return dis_min, id_min, dis_other

    def update_dis2collision(self, C_point):
        # update dis2collision of every waypoint when detect a new collision point
        self.collisionpoints.append(C_point)
        for i in range(len(self.waypoints)):
            distance = self.get_distance(self.waypoints[i]['pose'], C_point)
            self.waypoints[i]['dis2collision'] = min(self.waypoints[i]['dis2collision'], distance)

    def select_waypoint_distance(self, currentpose):
        # select the farthest point in history
        dis = dict()
        i = 0
        for wp in self.waypoints:
            dis[i] = self.get_distance(currentpose, wp['pose'])
            i += 1
        dis_list = sorted(dis.items(), key=lambda item: item[1], reverse=True)
        # random sample the pos
        start_id = random.randint(0, (len(dis_list) - 1) / 2)
        startpoint = self.waypoints[dis_list[start_id][0]]
        return startpoint

    def select_waypoint_random(self):
        self.start_id = random.randint(0, (len(self.waypoints) - 1))
        startpoint = self.waypoints[self.start_id]['pose']
        self.waypoints[self.start_id]['selected'] += 1
        return startpoint



