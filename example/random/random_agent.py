import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='RobotArm-Discrete-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", "--render", default=False, metavar='G', help='show env using cv2')
    args = parser.parse_args()
    env = gym.make(args.env_id)

    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        env.seed(i)
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            print ('reward: ' + str(reward))
            if args.render:
                img = env.render(mode='rgb_array')
                #  img = img[..., ::-1]  # bgr->rgb
                cv2.imshow('show', img)
                cv2.waitKey(1)
            if done:
                break

    # Close the env and write monitor result info to disk
    env.close()


