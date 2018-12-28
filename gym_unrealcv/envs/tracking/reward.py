import numpy as np
class Reward():
    '''
    define different type reward function
    '''

    def __init__(self, setting):

        self.dis2target_last = 0
        self.dis_exp = setting['exp_distance']
        self.dis_max = setting['max_distance']

    def reward_distance(self, dis2target_now, direction_error, dis_exp=None):
        #  reward = (100.0 / max(dis2target_now,100)) * np.cos(direction_error/360.0*np.pi)
        if dis_exp is None:
            dis_exp = self.dis_exp
        direction_error = abs(direction_error/45.0)
        e_dis = abs(dis_exp - dis2target_now)
        #  e_dis_relative = e_dis / self.dis_exp
        e_dis_relative = e_dis / dis_exp
        # reward = 1 - min(e_dis_relative, 1) - min(direction_error, 1)
        reward = 1 - direction_error - e_dis_relative
        self.dis2target_last = dis2target_now
        reward = max(reward, -1)
        return reward

    def reward_target(self, dis2target_now, direction_error, dis_exp=None):
        #  reward = (100.0 / max(dis2target_now,100)) * np.cos(direction_error/360.0*np.pi)
        if dis_exp is None:
            dis_exp = self.dis_exp
        direction_error = abs(abs(direction_error) - 45)/45.0
        e_dis = abs(abs(dis_exp - dis2target_now) - dis_exp) / dis_exp
        # reward = 1 - 2 * min(abs(e_dis_relative), 1) + min(abs(direction_error/(np.pi/4)), 2)
        # reward = 1 - min(e_dis, 1) - min(direction_error, 1)
        reward = 1 - e_dis - direction_error
        reward = max(reward, -1)
        self.dis2target_last = dis2target_now
        return reward
