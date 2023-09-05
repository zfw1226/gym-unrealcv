import gym
from gym import Wrapper

class ConfigUEWrapper(Wrapper):
    def __init__(self, env,  display=None, offscreen_rendering=False, use_opengl=False, nullrhi=False):
        super().__init__(env)
        env.unwrapped.display = display
        env.unwrapped.offscreen_rendering = offscreen_rendering
        env.unwrapped.use_opengl = use_opengl
        env.unwrapped.nullrhi = nullrhi

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self, **kwargs):
        states = self.env.reset(**kwargs)
        return states