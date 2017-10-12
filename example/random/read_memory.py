import json
import os
import cv2
import time


env_id = 'RobotArm-Discrete-v0'
eps = os.listdir(env_id)
print len(eps)
for ep in eps:
    # read
    #print ep
    ep_pwd = os.path.join(env_id,ep)
    info_pwd = os.path.join(ep_pwd,'info.json')
    if not os.path.exists(info_pwd):
        continue
    with open(info_pwd) as f :
        info_all = json.load(f)

    C_reward = 0
    for info_step in info_all:
        C_reward += info_step['reward']
        #if info_step['reward'] >5 :
        print ('Ep:{} reward:{}'.format(ep, info_step['reward']))
        '''
        img_pwd = os.path.join(ep_pwd,info_step['img_name'])
        img = cv2.imread(img_pwd)
        cv2.imshow('img',img)
        cv2.waitKey(10)
        time.sleep(0.1)
        '''
    if C_reward > 0:
        print ('Ep:{} C_reward:{}'.format(ep,C_reward))