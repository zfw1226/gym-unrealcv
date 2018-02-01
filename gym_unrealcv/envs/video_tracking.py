import os
import cv2
from gym import spaces
import gym
import numpy as np
from matplotlib import pyplot as plt
class VideoTracking_base(gym.Env):
    def __init__(self,
                 dataset_root='/data/zfw/VOTchallenge',
                 year = 'vot2015',
                 seq = 'crossing', # gymnastics, woman, sunshade, iceskater jogging
                 setting_file = 'tracking_v0.4_F.json',
                 ):
        self.dataset_dir = os.path.join(dataset_root,year,seq)
        self.img_list = os.listdir(self.dataset_dir)
        self.load_env_setting(setting_file)
        self.name = os.path.join(year,seq)
        self.skip_step = 1
        self.img_list.sort()
        self.action_space = spaces.Discrete(len(self.discrete_actions))
        state = cv2.imread(os.path.join(self.dataset_dir,self.img_list[0]))
        self.observation_space = spaces.Box(low=0, high=255., shape=state.shape)
        self.img_id = 0
        self.max_steps = len(self.img_list) - 10

        # load ground truth
        self.x_center = []
        self.y_center = []
        self.size = []


        with open(os.path.join(self.dataset_dir,'groundtruth.txt')) as f:
            for line in f:
                xy = line.split(',')

                if year == 'vot2013':
                    self.x_center.append(float(xy[0])+ float(xy[2])/2)
                    self.y_center.append(float(xy[1]) + float(xy[3]) / 2)
                    self.size.append(float(xy[2])*float(xy[3]))
                else:
                    x_sum = 0
                    y_sum = 0
                    for i in range(0,len(xy),2):
                        x_sum += float(xy[i])
                        y_sum += float(xy[i+1])
                    self.x_center.append(x_sum / 4)
                    self.y_center.append(y_sum / 4)
                    self.size.append(abs(float(xy[5])-float(xy[1]))  * abs(float(xy[0])- float(xy[2])))

        self.truth = []

    def _step(self,action):

        state = cv2.imread(os.path.join(self.dataset_dir,self.img_list[self.img_id]))

        h0 = 0
        h1 = state.shape[0]
        w0 = state.shape[1]/2 - (h1-h0)/2
        w1 = state.shape[1]/2 + (h1-h0)/2
        state = state[h0:h1,w0:w1]
        height = state.shape[0]
        width = state.shape[1]

        cv_img = state.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX



        action_x = int(5 * width / 10)
        action_y = int(9 * height / 10)
        color = [(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)]
        color[action] = (0, 0, 255)
        cv2.circle(cv_img, (action_x + int(width * 0.04), action_y - int(height * 0.2)), 8, color[0], -1)  # forward
        cv2.circle(cv_img, (action_x + int(width * 0.04), action_y - int(height * 0.15)), 8, color[1], -1)  # left
        cv2.circle(cv_img, (action_x + 10 + int(width * 0.04), action_y - int(height * 0.10)), 8, color[2], -1)  # right
        cv2.circle(cv_img, (action_x - 10 + int(width * 0.04), action_y - int(height * 0.10)), 8, color[3], -1)  # left
        cv2.circle(cv_img, (action_x + 20 + int(width * 0.04), action_y - int(height * 0.05)), 8, color[4], -1)  # right
        cv2.circle(cv_img, (action_x - 20 + int(width * 0.04), action_y - int(height * 0.05)), 8, color[5], -1)  # left
        cv2.circle(cv_img, (action_x + int(width * 0.04), action_y - int(height * 0.05)), 8, color[6], -1)  # left

        cv2.circle(cv_img, (int(self.x_center[self.img_id] - w0), int(self.y_center[self.img_id])), int(self.size[self.img_id]/300), (0, 255, 0), -1)  # target

        self.truth.append([action,self.size[self.img_id],self.x_center[self.img_id],self.y_center[self.img_id]])
        #print [self.size[self.img_id],self.x_center[self.img_id],self.y_center[self.img_id]]
        cv2.imshow('state',cv_img)
        cv2.waitKey(10)
        self.img_id += self.skip_step

        if action == 2 or action == 4:
            color = 'r'
        elif action == 3 or action == 5:
            color = 'g'
        elif action == 6:
            color = 'y'
        else:
            color = 'c'


        plt.scatter(self.x_center[self.img_id], self.size[self.img_id]/100.0, c=color, alpha=0.4, s=25,marker = 'o')

        if self.img_id >= self.max_steps:
            done = True
            np.save(self.name,np.array(self.truth))
            plt.show()
            '''
            for truth in self.truth:
                #print truth
                ave_size = np.mean(np.array(truth)[:,0])
                ave_x = np.mean(np.array(truth)[:,1])
                ave_y = np.mean(np.array(truth)[:, 2])
                print ave_size,ave_x-width/2, ave_y
            '''
        else:
            done = False
        reward = 0
        info = dict()
        return state,reward,done,info
    def _reset(self):
        self.img_id = 0
        state = cv2.imread(os.path.join(self.dataset_dir,self.img_list[self.img_id]))

        self.truth = []
        for i in range(len(self.discrete_actions)):
            self.truth.append([])
        return  state

    def load_env_setting(self, filename):
        f = open(self.get_settingpath(filename))
        type = os.path.splitext(filename)[1]
        if type == '.json':
            import json
            setting = json.load(f)
        elif type == '.yaml':
            import yaml
            setting = yaml.load(f)
        else:
            print 'unknown type'

        # print setting
        self.cam_id = setting['cam_id']
        self.target_list = setting['targets']
        self.max_steps = setting['max_steps']
        self.discrete_actions = setting['discrete_actions']
        self.continous_actions = setting['continous_actions']
        self.max_distance = setting['max_distance']
        self.max_direction = setting['max_direction']
        self.objects_env = setting['objects_list']

        return setting

    def get_settingpath(self, filename):
        import gym_unrealcv
        gympath = os.path.dirname(gym_unrealcv.__file__)
        return os.path.join(gympath, 'envs/setting', filename)