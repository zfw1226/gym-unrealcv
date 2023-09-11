import gym
from gym import Wrapper
import numpy as np

class RandomPopulationWrapper(Wrapper):
    def __init__(self, env,  num_min=2, num_max=10, random_target=False, random_tracker=False):
        super().__init__(env)
        self.min_num = num_min
        self.max_num = num_max
        self.random_target_id = random_target
        self.random_tracker_id = random_tracker

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self, **kwargs):
        env = self.env.unwrapped
        if not env.launched:  # we need to launch the environment
            env.launched = env.launch_ue_env()
            env.init_agents()
            env.init_objects()

        # Randomize the number of agents
        env.num_agents = np.random.randint(self.min_num, self.max_num)
        env.adjust_agents(env.num_agents)
        if self.random_tracker_id:
            env.tracker_id = env.sample_tracker()
        if self.random_target_id:
            env.target_id = env.sample_target()
        states = self.env.reset(**kwargs)

        return states