import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import time
import numpy as np
from gym_unrealcv.envs.wrappers import time_dilation, early_done, auto_monitor

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        # return 2
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrackingMPRoom-DiscreteColor-v2',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=0, help='random seed')
    parser.add_argument("-t", '--time_dilation', dest='time_dilation', default=10, help='time_dilation to keep fps in simulator')
    parser.add_argument("-d", '--early_done', dest='early_done', default=100, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')

    args = parser.parse_args()
    env = gym.make(args.env_id)
    if args.time_dilation > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, args.time_dilation)
    if args.early_done > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, args.early_done)
    if args.monitor:
        env = auto_monitor.DisplayWrapper(env)
    episode_count = 100
    rewards = 0
    done = False

    Total_rewards = 0
    env.seed(int(args.seed))
    try:
        for eps in range(1, episode_count):
            obs = env.reset()
            agents_num = len(env.action_space)
            agents = [RandomAgent(env.action_space[i]) for i in range(len(env.action_space))] # reset agents
            count_step = 0
            t0 = time.time()
            agents_num = len(obs)
            C_rewards = np.zeros(agents_num)
            while True:
                actions = [agents[i].act(obs[i]) for i in range(agents_num)]
                obs, rewards, done, _ = env.step(actions)
                C_rewards += rewards
                count_step += 1
                if args.render:
                    img = env.render(mode='rgb_array')
                    #  img = img[..., ::-1]  # bgr->rgb
                    cv2.imshow('show', img)
                    cv2.waitKey(1)
                if done:
                    fps = count_step/(time.time() - t0)
                    Total_rewards += C_rewards[0]
                    print ('Fps:' + str(fps), 'R:'+str(C_rewards), 'R_ave:'+str(Total_rewards/eps))
                    break

        # Close the env and write monitor result info to disk
        print('Finished')
        env.close()
    except KeyboardInterrupt:
        print('exiting')
        env.close()


