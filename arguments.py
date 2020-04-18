import os
import random
import argparse
import torch
import numpy as np

from utils import cleanup_log_dir


def get_args():
    parser = argparse.ArgumentParser(description='Batch_PPO')
    parser.add_argument('--task_id', type=str, default='HalfCheetahBulletEnv-v0',
                        help='task name (default: Pendulum-v0)')
    parser.add_argument('--run_id', type=str, default='no_norm_env_1',
                        help="name of the run")
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num_processes', type=int, default=12,
                        help='number of parallel processes (default: 12)')
    parser.add_argument("--disable_cuda", default=False, help='Disable CUDA')

    # Training config
    parser.add_argument('--num-env-steps', type=int, default=5e6,
                        help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='Optimizer epsilon (default: 1e-5)')
    parser.add_argument('--use-linear-lr-decay', type=bool, default=True,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--num-steps', type=int, default=2048,
                        help='number of forward environment steps (default: 1000)')

    # PPO config
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='reward discount coefficient (default: 0.99)')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')

    parser.add_argument('--ppo-epoch', type=int, default=10,
                        help='number of ppo epochs (default: 10)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of mini batches (default: 32)')

    # Log config
    parser.add_argument('--log-interval', type=int, default=1,
                        help='log interval, one log per n updates (default: 1)')
    parser.add_argument('--log-dir', type=str, default='log/',
                        help='directory to save agent logs (default: log/)')
    parser.add_argument('--monitor-dir', type=str, default='monitor_log/',
                        help='directory to save monitor logs (default: monitor_log/)')
    parser.add_argument('--result-dir', type=str, default='results/',
                        help='directory to save plot results (default: results/)')

    # Evaluate performance
    parser.add_argument('--test_iters', type=int, default=int(1e4),
                        help='test iterations (default: 1000)')
    parser.add_argument('--video_width', type=int, default=720,
                        help='video resolution (default: 720)')
    parser.add_argument('--video_height', type=int, default=720,
                        help='video resolution (default: 720)')

    # Saving and restoring setup
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--save-dir', type=str, default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')

    args = parser.parse_args()

    # Create directories
    args.save_path = os.path.join("saves", args.task_id, args.run_id)
    os.makedirs(args.save_path, exist_ok=True)
    args.result_dir = os.path.join(args.result_dir, args.task_id)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    cleanup_log_dir(args.log_dir)

    args.monitor_dir = os.path.join(args.monitor_dir, args.task_id, args.run_id)
    os.makedirs(args.monitor_dir, exist_ok=True)

    # Setup device and random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
    else:
        args.device = torch.device('cpu')
    torch.set_num_threads(1)

    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

    return args


if __name__ == '__main__':
    args = get_args()
