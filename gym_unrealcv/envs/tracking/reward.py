import numpy as np
class Reward():
    '''
    define different type reward function
    '''

    def __init__(self, setting):

        self.dis2target_last = 0
        self.dis_exp = 150


    def reward_distance(self, dis2target_now, direction_error):
        #reward = (100.0 / max(dis2target_now,100)) * np.cos(direction_error/360.0*np.pi)
        direction_error  = direction_error/180.0*np.pi
        cos_factor = np.cos(direction_error)

        reward = 1 - abs(self.dis_exp-dis2target_now) / self.dis_exp * (1 - cos_factor) - abs(direction_error/np.pi)
        self.dis2target_last = dis2target_now
        #print reward,direction_error
        return reward

    def reward_move(self):
        return 10
