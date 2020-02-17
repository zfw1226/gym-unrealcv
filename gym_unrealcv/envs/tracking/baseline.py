from gym_unrealcv.envs.utils import misc
import numpy as np


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.step_counter = 0
        self.keep_steps = 0
        self.action_space = action_space

    def act(self, pose):
        self.step_counter += 1
        if self.pose_last == None:
            self.pose_last = pose
            d_moved = 100
        else:
            d_moved = np.linalg.norm(np.array(self.pose_last) - np.array(pose))
            self.pose_last = pose
        if self.step_counter > self.keep_steps or d_moved < 3:
            self.action = self.action_space.sample()
            if self.action == 1 or self.action == 6 or self.action == 0:
                self.action = 0
                self.keep_steps = np.random.randint(10, 20)
            elif self.action == 2 or self.action == 3:
                self.keep_steps = np.random.randint(1, 20)
            else:
                self.keep_steps = np.random.randint(1, 10)
        return self.action

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0
        self.pose_last = None


class GoalNavAgent(object):
    def __init__(self, action_space, goal_area, nav):
        self.step_counter = 0
        self.keep_steps = 0
        self.velocity_high = action_space['high'][0]
        self.velocity_low = action_space['low'][0]
        self.angle_high = action_space['high'][1]
        self.angle_low = action_space['low'][1]
        self.goal_area = goal_area
        # self.goal = self.generate_goal(self.goal_area)
        if 'Base' in nav:
            self.discrete = True
        else:
            self.discrete = False
        if 'Short' in nav:
            self.max_len = 30
        elif 'Mid' in nav:
            self.max_len = 100
        else:
            self.max_len = 1000
        if 'Fix' in nav:
            self.fix = True
        else:
            self.fix = False

    def act(self, pose):
        self.step_counter += 1
        if self.pose_last == None or self.fix:
            self.pose_last = pose
            d_moved = 100
        else:
            d_moved = np.linalg.norm(np.array(self.pose_last) - np.array(pose))
            self.pose_last = pose
        if self.check_reach(self.goal, pose) or d_moved < 3 or self.step_counter > self.max_len:
            self.goal = self.generate_goal(self.goal_area, self.fix)
            if self.discrete or self.fix:
                self.velocity = (self.velocity_high + self.velocity_low)/2
            else:
                self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
            # self.velocity = 70
            self.step_counter = 0

        delt_yaw = misc.get_direction(pose, self.goal)
        if self.discrete:
            if abs(delt_yaw) > self.angle_high:
                velocity = 0
            else:
                velocity = self.velocity
            if delt_yaw > 3:
                self.angle = self.angle_high / 2
            elif delt_yaw < -3:
                self.angle = self.angle_low / 2
        else:
            self.angle = np.clip(delt_yaw, self.angle_low, self.angle_high)
            velocity = self.velocity * (1 + 0.2*np.random.random())
        return (velocity, self.angle)

    def act2(self, pose):
        if self.pose_last == None or self.fix:
            self.pose_last = pose
            d_moved = 100
        else:
            d_moved = np.linalg.norm(np.array(self.pose_last) - np.array(pose))
            self.pose_last = pose
        if d_moved < 10:
            self.step_counter += 1
        if self.step_counter > 3:
            self.goal = self.generate_goal(None, self.fix)
            self.velocity = (self.velocity_high + self.velocity_low) / 2
            self.step_counter = 0
            return (self.velocity, 0), self.goal
        else:
            return (0, 0), None

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0
        self.goal_id = 0
        self.goal = self.generate_goal(self.goal_area, self.fix)
        self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
        self.pose_last = None

    def generate_goal(self, goal_area, fixed=False):
        if goal_area==None:
            goal_area = self.goal_area
        goal_list = [[goal_area[0], goal_area[2]], [goal_area[0], goal_area[3]],
                     [goal_area[1], goal_area[3]], [goal_area[1], goal_area[2]]]
        np.random.seed()
        if fixed:
            goal = np.array(goal_list[self.goal_id % len(goal_list)])/2
            self.goal_id += 1
        else:
            x = np.random.randint(goal_area[0], goal_area[1])
            y = np.random.randint(goal_area[2], goal_area[3])
            goal = np.array([x, y])
        return goal

    def check_reach(self, goal, now):
        error = np.array(now[:2]) - np.array(goal[:2])
        distance = np.linalg.norm(error)
        return distance < 50

class GoalNavAgentTest(object):
    def __init__(self, action_space, goal_list=None):
        self.step_counter = 0
        self.keep_steps = 0
        self.goal_id = 0
        self.velocity_high = action_space['high'][0]
        self.velocity_low = action_space['low'][0]
        self.angle_high = action_space['high'][1]
        self.angle_low = action_space['low'][1]
        self.goal_list = goal_list

        self.goal = self.generate_goal()
        self.discrete = False
        self.max_len = 1000

    def act(self, pose):

        self.step_counter += 1
        if self.pose_last == None:
            self.pose_last = pose
            d_moved = 100
        else:
            d_moved = np.linalg.norm(np.array(self.pose_last) - np.array(pose))
            self.pose_last = pose
        if self.check_reach(self.goal, pose) or d_moved < 3 or self.step_counter > self.max_len:
            self.goal = self.generate_goal()
            if self.discrete:
                self.velocity = (self.velocity_high + self.velocity_low) / 2
            else:
                self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
            self.step_counter = 0

        delt_yaw = self.get_direction(pose, self.goal)
        if self.discrete:
            if abs(delt_yaw) > self.angle_high:
                velocity = 0
            else:
                velocity = self.velocity
            if delt_yaw > 3:
                self.angle = self.angle_high / 2
            elif delt_yaw < -3:
                self.angle = self.angle_low / 2
        else:
            self.angle = np.clip(delt_yaw, self.angle_low, self.angle_high)
            velocity = self.velocity * (1 + 0.2 * np.random.random())

        return (velocity, self.angle)

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0
        self.goal_id = 0
        self.goal = self.generate_goal()
        self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
        self.pose_last = None

    def generate_goal(self):
        index = self.goal_id % len(self.goal_list)
        goal = np.array(self.goal_list[index])

        self.goal_id += 1
        return goal

    def check_reach(self, goal, now):
        error = np.array(now[:2]) - np.array(goal[:2])
        distance = np.linalg.norm(error)
        return distance < 50

    def get_direction(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        angle_now = np.arctan2(y_delt, x_delt) / np.pi * 180 - current_pose[4]
        if angle_now > 180:
            angle_now -= 360
        if angle_now < -180:
            angle_now += 360
        return angle_now