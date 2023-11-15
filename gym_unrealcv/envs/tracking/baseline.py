import random
import gym
from gym_unrealcv.envs.utils import misc
import numpy as np
from random import choice

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space, min_steps=1, max_steps=20):
        self.step_counter = 0
        self.keep_steps = 0
        self.action_space = action_space
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.reset()

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
            self.keep_steps = np.random.randint(self.min_steps, self.max_steps)
            # if self.action == 1 or self.action == 6 or self.action == 0:
            #     self.action = 0
            #     self.keep_steps = np.random.randint(10, 20)
            # elif self.action == 2 or self.action == 3:
            #     self.keep_steps = np.random.randint(1, 20)
            # else:
            #     self.keep_steps = np.random.randint(1, 10)
        return self.action

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0
        self.pose_last = None


class GoalNavAgent(object):
    def __init__(self, action_space, goal_area, nav, random_th=0):
        self.step_counter = 0
        self.keep_steps = 0
        self.velocity_high = action_space['high'][0]
        self.velocity_low = action_space['low'][0]
        self.angle_high = action_space['high'][1]
        self.angle_low = action_space['low'][1]
        self.goal_area = goal_area
        self.random_th = random_th
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

    def act(self, pose, ref_goal=None):
        self.step_counter += 1
        if self.pose_last == None or self.fix:
            self.pose_last = pose
            d_moved = 100
        else:
            d_moved = np.linalg.norm(np.array(self.pose_last) - np.array(pose))
            self.pose_last = pose
        if self.check_reach(self.goal, pose) or d_moved < 3 or self.step_counter > self.max_len:
            if ref_goal is None or np.random.random() > self.random_th:
                self.goal = self.generate_goal(self.goal_area, self.fix)
            else:
                self.goal = ref_goal
            self.step_counter = 0

        if np.random.random() < 0.05:
            self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
        if np.random.random() < 0.01 and self.angle_noise_step == 0:  # noise angle
            self.angle = choice([1, -1])*self.angle_high*(1 + 0.2*np.random.random())
            self.angle_noise_step = np.random.randint(5, 20)
        else:
            self.angle_noise_step = 0

        delt_yaw = misc.get_direction(pose, self.goal) # get the angle between current pose and goal in x-y plane
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
            if self.angle_noise_step > 0:
                angle = self.angle
                self.angle_noise_step -= 1
            else:
                angle = np.clip(delt_yaw, self.angle_low, self.angle_high)
            velocity = self.velocity * (1 + 0.2*np.random.random())
        return (velocity, angle)

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
        self.angle_noise_step = 0
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

    def check_reach(self, goal, pose_now, dim=2):
        error = np.array(pose_now[:dim]) - np.array(goal[:dim])
        distance = np.linalg.norm(error)
        return distance < 50

class InternalNavAgent(object):
    def __init__(self, goal_list=None, goal_area=None, max_len=100):
        self.step_counter = 0
        self.keep_steps = 0
        self.goal_area = goal_area
        self.goal_id = 0
        self.goal_list = goal_list

        self.max_len = max_len
        self.goal = self.reset()

    def act(self, pose):
        self.step_counter += 1
        self.pose_last = pose
        if self.check_reach(self.goal, pose) or self.step_counter > self.max_len:
            # sample a new goal
            self.goal = self.generate_goal()
            self.step_counter = 0
            return self.goal
        else:
            return None

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0
        self.goal_id = 0
        self.goal = self.generate_goal()
        self.pose_last = None
        return self.goal

    def sample_goal_from_list(self, is_random=False):
        if is_random:
            goal = random.choice(self.goal_list)
            self.goal_id = self.goal_list.index(goal)
        else:
            index = self.goal_id % len(self.goal_list)
            goal = np.array(self.goal_list[index])
            self.goal_id += 1
        return goal

    def generate_goal(self, goal_area=None, fixed=False):
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
            z = np.random.randint(goal_area[4], goal_area[5])
            goal = np.array([x, y, z])
        return goal

    def check_reach(self, goal, pose_now, dim=2):
        error = np.array(pose_now[:dim]) - np.array(goal[:dim])
        distance = np.linalg.norm(error)
        return distance < 50

class Nav2GoalAgent(object):
    def __init__(self, action_space, goal_area, fix_point=False, random_th=0, max_len=200):
        self.step_counter = 0
        self.keep_steps = 0
        if type(action_space) == gym.spaces.Discrete:
            self.discrete = True
        else:
            self.discrete = False
            self.velocity_high = action_space.high[1]
            self.velocity_low = 0
            self.angle_high = action_space.high[0]
            self.angle_low = action_space.low[0]
        self.goal_area = goal_area
        self.random_th = random_th

        self.max_len = max_len
        self.fix = fix_point

        self.reset()

    def act(self, pose, ref_goal=None):
        self.step_counter += 1
        if self.pose_last == None or self.fix:
            self.pose_last = pose
            d_moved = 10
        else:
            d_moved = np.linalg.norm(np.array(self.pose_last) - np.array(pose)) # get the distance moved for checking if the agent is stuck
            self.pose_last = pose
        self.d_move_ave = self.d_move_ave*0.7 + d_moved*0.3
        if self.check_reach(self.goal, pose) or self.d_move_ave < 3 or self.step_counter > self.max_len:
            self.d_move_ave = 5
            if ref_goal is None or np.random.random() > self.random_th:
                self.goal = self.generate_goal(self.goal_area, self.fix)
                self.velocity = np.random.randint(0.5*self.velocity_high, self.velocity_high)
            else:
                self.goal = ref_goal
            self.step_counter = 0

        delt_yaw = misc.get_direction(pose, self.goal) # get the angle between current pose and goal in x-y plane
        angle = np.clip(delt_yaw, self.angle_low, self.angle_high)
        velocity = self.velocity * (1 + 0.2*np.random.random())
        return [angle, velocity]

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0
        self.angle_noise_step = 0
        self.goal_id = 0
        self.d_move_ave = 5
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

class PoseTracker(object):
    def __init__(self, action_space, expected_distance = 250, expected_angle = 0):
        if type(action_space) == gym.spaces.Discrete:
            self.discrete = True
        else:
            self.discrete = False
            self.velocity_high = action_space.high[1]
            self.velocity_low = action_space.low[1]
            self.angle_high = action_space.high[0]
            self.angle_low = action_space.low[0]
        self.expected_distance = expected_distance
        self.expected_angle = expected_angle
        from simple_pid import PID
        self.angle_pid = PID(1, 0.01, 0, setpoint=1)
        self.velocity_pid = PID(5, 0.1, 0.05, setpoint=1)

    def act(self, pose, target_pose):
        delt_yaw = misc.get_direction(pose, target_pose) # get the angle between current pose and goal in x-y plane
        angle = np.clip(self.angle_pid(-delt_yaw), self.angle_low, self.angle_high)
        delt_distance = (np.linalg.norm(np.array(pose[:2]) - np.array(target_pose[:2])) - self.expected_distance)
        velocity = np.clip(self.velocity_pid(-delt_distance), self.velocity_low, self.velocity_high)
        return [angle, velocity]