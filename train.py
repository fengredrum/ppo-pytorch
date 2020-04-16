import os
import time
from collections import deque

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from arguments import get_args
from environment import make_vec_envs
from model import Policy
from ppo import PPO
from storage import ReplayBuffer
from utils import update_linear_schedule


def main(args, idx):
    # Create summary writer
    writer_path = os.path.join(args.log_dir, args.task_id, args.run_id + '-' + str(idx))
    writer = SummaryWriter(log_dir=writer_path)

    # Create training envs
    envs = make_vec_envs(args.task_id, args.seed, args.num_processes,
                         args.gamma, args.log_dir, args.device)
    obs_size = envs.observation_space.shape[0]
    act_size = envs.action_space.shape[0]

    # Create NN
    actor_critic = Policy(obs_size, act_size,
                          action_range=[envs.action_space.low[0], envs.action_space.high[0]])
    actor_critic.to(args.device)

    # Create ppo agent
    agent = PPO(
        actor_critic=actor_critic,
        device=args.device,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        clip_param=args.clip_param,
        ppo_epoch=args.ppo_epoch,
        num_mini_batch=args.num_mini_batch,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
    )

    # Create replay buffer
    buffer = ReplayBuffer(args.num_steps, args.num_processes, obs_size, act_size)
    buffer.to(args.device)

    # Reset envs
    obs = envs.reset()
    buffer.obs[0].copy_(obs)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    for j in tqdm(range(num_updates)):

        if args.use_linear_lr_decay:
            update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        # Collect trajectories and compute returns
        with torch.no_grad():
            for step in range(args.num_steps):
                # Sample actions
                action = actor_critic(buffer.obs[step])

                # Get trajectories from envs
                obs, reward, done, infos = envs.step(action)
                mask = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done],
                    dtype=torch.float, device=args.device)
                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])

                # Store trajectories
                buffer.insert(obs, action, reward, mask)

            # Compute returns
            batch_obs = buffer.obs.view(-1, obs_size)
            value = actor_critic.get_value(batch_obs).view(args.num_steps + 1, args.num_processes, 1)
            batch_obs = buffer.obs[:-1].view(-1, obs_size)
            batch_action = buffer.actions.view(-1, act_size)
            action_log_prob = actor_critic.get_act_log_prob(batch_obs, batch_action).view(args.num_steps,
                                                                                          args.num_processes, 1)
            buffer.update_value_log_prob(value, action_log_prob)
            buffer.compute_returns(args.gamma, args.gae_lambda)

        # Update policy
        agent_output = agent.update(buffer)
        buffer.after_update()

        # Log stuff
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            speed = int(total_num_steps / (end - start))
            print(
                "Updates {}, num timesteps {}, FPS {} \n "
                "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, "
                "min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            speed,
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards),
                            ))
            writer.add_scalar('mean_reward', np.mean(episode_rewards), total_num_steps)
            writer.add_scalar('speed', speed, total_num_steps)
            for key in agent_output.keys():
                writer.add_scalar(key, agent_output[key], total_num_steps)

            if args.task_id == 'Pendulum-v0' and np.mean(episode_rewards) > -250:
                break

    envs.close()
    writer.close()


if __name__ == "__main__":
    from warnings import simplefilter

    simplefilter(action='ignore', category=FutureWarning)

    args = get_args()
    for i in range(1):
        main(args, i)
