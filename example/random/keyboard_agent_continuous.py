import argparse
import random

import gym_unrealcv
import gym
from gym import wrappers
import cv2
import time
import numpy as np
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation
from gym_unrealcv.envs.tracking.baseline import PoseTracker
from PIL import Image
import os
import torch
import threading
from queue import Queue
from pynput import keyboard
import time
tracker_speed=0.0
tracker_angle=0.0
target_speed=0.0
target_angle=0.0
speed_tmp=0.0
angle_tmp=0.0
keys=set()
tracker_key_angle=Queue(5)
tracker_key_speed=Queue(5)
target_key_angle=Queue(5)
target_key_speed=Queue(5)
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()
def increase_tracker_speed():
    global tracker_speed
    while 'i' in keys   and tracker_speed < 150:
        tracker_speed += 10
        tracker_key_speed.put(tracker_speed)
        time.sleep(0.2)
    while 'i' not in keys and tracker_speed > 0:
        tracker_speed  =0
        time.sleep(0.2)

def increase_target_speed():
    global target_speed
    while 't' in keys and target_speed < 150:
        target_speed += 10
        target_key_speed.put(target_speed)
        time.sleep(0.2)
    while 't' not in keys and target_speed > 0:
        target_speed = 0
        time.sleep(0.2)

def decrease_tracker_speed():
    global tracker_speed
    while 'k' in keys and tracker_speed > -150:
        tracker_speed -= 10
        tracker_key_speed.put(tracker_speed)
        time.sleep(0.1)
        break
    while 'k' not in keys and tracker_speed < 0:
        tracker_speed =0
        time.sleep(0.1)
def decrease_target_speed():
    global target_speed
    while 'g' in keys and target_speed > -150:
        target_speed -= 10
        target_key_speed.put(target_speed)
        time.sleep(0.1)
        break
    while 'g' not in keys and target_speed < 0:
        target_speed =0
        time.sleep(0.1)

def decrease_tracker_angle():
    global tracker_angle
    while 'j' in keys  and tracker_angle > -50:
        tracker_angle -= 5
        tracker_key_angle.put(tracker_angle)
        time.sleep(0.1)
    while 'j' not in keys and tracker_angle < 0:
        tracker_angle =0
        time.sleep(0.1)

def decrease_target_angle():
    global target_angle
    while 'f' in keys and target_angle > -50:
        target_angle -= 5
        target_key_angle.put(target_angle)
        time.sleep(0.1)
    while 'f' not in keys and target_angle < 0:
        target_angle = 0
        time.sleep(0.1)
    # print(angle)

def increase_tracker_angle():
    global tracker_angle
    while 'l' in keys and tracker_angle < 50:
        tracker_angle += 5
        tracker_key_angle.put(tracker_angle)
        time.sleep(0.1)
    while 'l' not in keys and tracker_angle > 0:
        tracker_angle =0
        time.sleep(0.1)

def increase_target_angle():
    global target_angle
    while 'h' in keys and target_angle < 50:
        target_angle += 5
        target_key_angle.put(target_angle)
        time.sleep(0.1)
    while 'h' not in keys and target_angle > 0:
        target_angle =0
        time.sleep(0.1)
def on_press(key):
    try: k = key.char # single-char keys
    except: k = key.name # other keys
    if k in ['i', 'j', 'k', 'l','t','f','g','h']:
        keys.add(k)
        if k == 'i':
            threading.Thread(target=increase_tracker_speed).start()
        elif k == 'k':
            threading.Thread(target=decrease_tracker_speed).start()
        elif k == 'j':
            threading.Thread(target=decrease_tracker_angle).start()
        elif k == 'l':
            threading.Thread(target=increase_tracker_angle).start()
        elif k=='t':
            threading.Thread(target=increase_target_speed).start()
        elif k=='g':
            threading.Thread(target=increase_target_speed).start()
        elif k=='f':
            threading.Thread(target=decrease_target_angle).start()
        elif k=='h':
            threading.Thread(target=increase_target_angle).start()



def on_release(key):
    try: k = key.char # single-char keys
    except: k = key.name # other keys
    if k in ['i', 'j', 'k', 'l']:
        keys.remove(k)

def get_key():
    global speed_tmp
    global angle_tmp
    while True:
        if not tracker_key_speed.empty():
            speed_tmp = tracker_key_speed.get()
            print('speed_tmp:',speed_tmp)
        if not tracker_key_angle.empty():
            angle_tmp = tracker_key_angle.get()
            print('angle_tmp:',angle_tmp)

        time.sleep(0.1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrackingMPRoom-DiscreteColor-v2',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=0, help='random seed')
    parser.add_argument("-p", '--path', default='./FlexibleRoom_Continuous/expert_480px_v4_Rgbd/', help='path to save the data')
    parser.add_argument("-t", '--time_dilation', dest='time_dilation', default=10, help='time_dilation to keep fps in simulator')
    parser.add_argument("-d", '--early_done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
    args = parser.parse_args()
    env = gym.make(args.env_id)
    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, args.time_dilation)
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, args.early_done)
    if args.monitor:
        env = monitor.DisplayWrapper(env)

    env = augmentation.RandomPopulationWrapper(env, 8, 10, random_target=True)
    env = agents.NavAgents(env, mask_agent=True)
    episode_count =100
    rewards = 0
    done = False

    Total_rewards = 0
    env.seed(int(args.seed))
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    get_queue = threading.Thread(target=get_key)
    get_queue.start()
    try:
        for eps in range(1, episode_count):
            obs = env.reset()
            agents_num = len(env.action_space)
            tracker_id = env.unwrapped.tracker_id
            target_id = env.unwrapped.target_id
            tracker = PoseTracker(env.action_space[0], env.unwrapped.exp_distance)  # TODO support multi trackers
            tracker_random=RandomAgent(env.action_space[0])

            count_step = 0
            t0 = time.time()
            agents_num = len(obs)
            C_rewards = np.zeros(agents_num)
            print('eps:', eps, 'agents_num:', agents_num)
            image = []
            action = []
            reward = []
            flag = 0
            pos = []
            print('start control!')
            while True:
                #pid tracker
                obj_poses = env.unwrapped.obj_poses
                actions = [tracker.act(obj_poses[tracker_id], obj_poses[target_id])]

                #random step actions
                flag -= 1
                if random.random() < 0.1 or flag > 0:
                    actions[0] = tracker_random.act(obj_poses)
                    # smooth action
                    if flag <= 0:
                        # action_tmp=actions[0]
                        flag = random.randint(1, 4)


                actions=[[np.clip(actions[0][0]+angle_tmp, tracker.angle_low, tracker.angle_high),
                          np.clip(actions[0][1]+speed_tmp, tracker.velocity_low, tracker.velocity_high)]]
                action.append(np.array(actions))
                obs, rewards, done, info = env.step(actions)
                C_rewards += rewards
                count_step += 1

                Depth_ob = obs[0][:,:,-1]

                # 将缩放后的图像数据添加到scaled_images中
                image.append(obs[0])
                pos.append(info['Pose_Obs'][0][1])
                reward.append(rewards)
                if args.render:
                    img = env.render(mode='rgb_array')
                    cv2.imshow('render',img.astype(np.uint8))
                    cv2.waitKey(1)
                if done:
                    fps = count_step / (time.time() - t0)
                    Total_rewards += C_rewards[0]
                    print('Fps:' + str(fps), 'R:' + str(C_rewards), 'R_ave:' + str(Total_rewards / eps))
                    is_first = np.array([True] + (len(image) - 1) * [False])
                    is_last = np.array((len(image) - 1) * [False] + [True])
                    dict = {
                        'action': action,
                        'image': image,
                        # 'pos':pos,
                        'reward': reward,
                        'is_first': is_first,
                        'is_last': is_last,

                    }
                    if args.path is not None:
                        if not os.path.exists(args.path):
                            os.makedirs(args.path)
                        save_dir = os.path.join(args.path, 'expert_480px_v4_' + "%04d" % int(eps+25) + "-%03d" % count_step + '.pt')
                        print('save data!')
                        torch.save(dict, save_dir)
                    break

        # Close the env and write monitor result info to disk
        print('Finished')
        env.close()
    except KeyboardInterrupt:
        print('exiting')
        env.close()

