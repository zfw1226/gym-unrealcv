import time

import gym
from gym import Wrapper
import numpy as np
from gym_unrealcv.envs.tracking.baseline import RandomAgent, Nav2GoalAgent, InternalNavAgent

class NavAgents(Wrapper):
    def __init__(self, env,  mask_agent = True):
        super().__init__(env)
        self.nav_list = []
        self.agents = []
        self.mask_agent = mask_agent

    def step(self, action):
        # the action is a list of actions for each agent, the length of the action is the number of agents
        env = self.env.unwrapped

        new_action = []
        for idx, mode in enumerate(self.nav_list):
            if mode == -1:
                if self.mask_agent:
                    new_action.append(action.pop(0))
                else:
                    new_action.append(action[idx])
                continue
            elif mode == 0:
                new_action.append(self.agents[idx].act(env.obj_poses[idx]))
            elif mode == 1:
                goal = self.agents[idx].act(env.obj_poses[idx])
                if goal is not None:
                    env.unwrapped.unrealcv.move_to(env.player_list[idx], goal)
                    # env.unwrapped.unrealcv.set_speed(env.player_list[idx], 200)
                new_action.append(None)
            elif mode == 2:
                new_action.append(self.agents[idx].act(env.obj_poses[idx]))
        obs, reward, done, info = self.env.step(new_action)
        obs = np.array([obs[i] for i in self.nav_list if i < 0])
        reward = np.array([reward[i] for i in self.nav_list if i < 0])

        return obs, reward, done, info

    def reset(self, **kwargs):
        states = self.env.reset(**kwargs)
        env = self.env.unwrapped

        # set nav list
        self.nav_list = []
        for i, obj_name in enumerate(env.player_list):
            '''
            config the navigation mode, 
            -1: directly pass the action to the agent 
            0: random
            1: use the internal navigation
            2: use the goal navigation
            '''
            if i == env.tracker_id:
                self.nav_list.append(-1)
            elif env.agents[obj_name]['agent_type'] in ['car', 'player']:
                self.nav_list.append(1)
            elif env.agents[obj_name]['agent_type'] in ['drone']:
                self.nav_list.append(0)
            else:
                self.nav_list.append(2)
            # print(f'{obj_name} use mode: {self.nav_list[-1]}')

        # init agents
        self.agents = []
        for idx, mode in enumerate(self.nav_list):
            if mode == -1:
                self.agents.append(None)
            elif mode == 0:
                self.agents.append(RandomAgent(env.action_space[i], 10, 50))
            elif mode == 1:
                self.agents.append(InternalNavAgent(env.safe_start, env.reset_area))
            elif mode == 2:
                self.agents.append(Nav2GoalAgent(env.action_space[i], env.reset_area, max_len=200))

        if self.mask_agent:
            states = np.array([states[i] for i in self.nav_list if i < 0])
            self.action_space = [self.env.action_space[i] for i in self.nav_list if i < 0]
            self.observation_space = [self.env.observation_space[i] for i in self.nav_list if i < 0]
        return states