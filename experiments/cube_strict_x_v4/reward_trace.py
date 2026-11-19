"""Record the task reward that reaches each CPU worker before VecNormalize."""
import gymnasium as gym

from experiments.cube_two_driver_v3.env import training_reward as baseline_reward
from .env import training_reward as strict_reward


class TaskRewardTrace(gym.Wrapper):
    def __init__(self, native, arm):
        reward_wrapper = {'v3': baseline_reward, 'v4': strict_reward}[arm]
        super().__init__(reward_wrapper(native))
        self.raw_return = 0.

    def __getattr__(self, name):
        # The native recorder reads MuJoCo data, the monitor and its configuration.
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self.unwrapped, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.raw_return = 0.
        return obs, {**info, 'raw_task_reward': None, 'raw_task_return': self.raw_return}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.raw_return += reward
        return obs, reward, terminated, truncated, {
            **info, 'raw_task_reward': float(reward), 'raw_task_return': float(self.raw_return),
        }


def record_task_rewards(env, arm):
    if isinstance(env, TaskRewardTrace):
        return env
    return TaskRewardTrace(env, arm)
