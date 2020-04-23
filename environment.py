import os
import gym
import numpy as np
import torch

from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv

try:
    import pybullet_envs
except ImportError:
    pass


def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = GymToFloat32(env)
        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  ):
    envs = [
        make_env(env_name, seed, i, log_dir)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    # if len(envs.observation_space.shape) == 1:
    #     if gamma is None:
    #         envs = VecNormalize(envs, ret=False)
    #     else:
    #         envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    return envs


class GymToFloat32(gym.ObservationWrapper):

    def __init__(self, env):
        super(GymToFloat32, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=old_shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        if observation.dtype != np.float32:
            return observation.astype(np.float32)
        else:
            return observation


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float().to(self.device)
        return obs, reward, done, info


if __name__ == '__main__':
    from arguments import get_args

    args = get_args()
    envs = make_vec_envs(args.task_id, args.seed, args.num_processes,
                         args.gamma, args.log_dir, args.device)

    obs = envs.reset()
    print('obs: ', obs.shape)
    print('low: ', envs.action_space.low[0])
    print('high: ', envs.action_space.high[0])
