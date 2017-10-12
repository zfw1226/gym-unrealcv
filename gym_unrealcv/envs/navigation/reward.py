class Reward():
    '''
    define different type reward function
    '''
    def __init__(self, setting):
        self.reward_factor = setting['reward_factor']
        self.reward_th = setting['reward_th']
        self.dis2target_last = 0


    def reward_bbox(self, boxes):
        reward = 0
        for box in boxes:
            reward += self.get_bbox_reward(box)  #sum the reward of all detected boxes

        if reward > self.reward_th:
            reward = min(reward * self.reward_factor, 10)
            print ('Get ideal Target!!!')
        elif reward == 0:
            reward = -1
            print ('Get Nothing')
        else:
            reward = 0
            print ('Get small Target!!!')

        return reward, boxes


    def get_bbox_reward(self, box):  # get reward of single box considering the size and position of box
        (xmin, ymin), (xmax, ymax) = box
        boxsize = (ymax - ymin) * (xmax - xmin)
        x_c = (xmax + xmin) / 2.0
        x_bias = x_c - 0.5
        discount = max(0, 1 - x_bias ** 2)
        reward = discount * boxsize
        return reward

    def reward_distance(self, dis2target_now):
        reward = (self.dis2target_last - dis2target_now) / max(self.dis2target_last, 100)
        self.dis2target_last = dis2target_now

        return reward

    def reward_move(self):
        return 10
