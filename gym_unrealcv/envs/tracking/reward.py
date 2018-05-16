import numpy as np
class Reward():
    '''
    define different type reward function
    '''

    def __init__(self, setting):

        self.dis2target_last = 0
        self.dis_exp = setting['exp_distance']

    def reward_distance(self, dis2target_now, direction_error, dis_exp=None):
        #  reward = (100.0 / max(dis2target_now,100)) * np.cos(direction_error/360.0*np.pi)
        if dis_exp is None:
            dis_exp = self.dis_exp
        direction_error = direction_error/180.0*np.pi

        e_dis = dis_exp-dis2target_now * np.cos(direction_error)
        #  e_dis_relative = e_dis / self.dis_exp
        e_dis_relative = e_dis / dis_exp
        reward = 1 - min(abs(e_dis_relative), 2) - min(abs(direction_error/(np.pi/4)), 2)
        self.dis2target_last = dis2target_now
        reward = max(reward, -1)
        return reward
