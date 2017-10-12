import argparse
import gym_unrealcv
import gym
from gym import wrappers
from example.utils import preprocessing
import json
import os
import cv2
import numpy as np

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e","--env_id", nargs='?', default='Search-RrMultiPlantsDiscreteTest-v0', help='Select the environment to run')
    args = parser.parse_args()
    env = gym.make(args.env_id)

    process_img = preprocessing.preprocessor(observation_space=env.observation_space, length=3,
                                             size=(100, 100))


    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)



    C_steps = 0
    episode_count = 100
    reward = 0
    done = False
    folderhead = os.path.join(args.env_id,'ep')

    if not os.path.exists(args.env_id):
        os.makedirs(args.env_id)
    init_eps = len(os.listdir(args.env_id))
    print init_eps

    for i in range(episode_count):
        ob = env.reset()
        eps = []
        foldername=folderhead+str(i+init_eps)
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        #print ob.shape
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, info = env.step(action)


            step = dict(
                img_name = str(info['Steps'])+'.jpg',
                arm_pose=info['ArmPose'],
                target_pose = info['TargetPose'],
                grip_position = info['GripPosition'],
                reward=info['Reward'],
                action=info['Action'],
                bbox=info['Bbox'],
                collision=info['Collision']
            )
            img_path=os.path.join(foldername,step['img_name'])
            cv2.imwrite(img_path,ob)
            eps.append(step)

            if done:
                C_steps += info['Steps']
                filename = 'info.json'
                filename = os.path.join(foldername,filename)
                with open(filename, 'w') as f:
                    json.dump(eps, f)
                break

    # Close the env and write monitor result info to disk
    env.close()


