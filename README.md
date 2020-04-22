# ppo-pytorch

This is a PyTorch implementation of PPO algorithm, 
which is designed for flexible modification and high performance 
in continuous control tasks.

## Requirements 

- pytorch 1.4.0
- tensorboard
- numpy
- tqdm 
- gym
- baselines
- pybullet (optional)

## Setup

You can use the provided `requirements.txt` file to install necessary dependencies.

```bash
$ pip install -r requirements.txt
```

## Experiment Results

It takes about half an hour for $5e6$ training steps in a six cores MacBook Pro.

HalfCheetahBulletEnv |  AntBulletEnv
:----------------------------------------------------------------------------:|:-----------------------------------------:
<img src="results/HalfCheetahBulletEnv-v0/baseline.png" alt="HalfCheetahBulletEnv" width="500px"> | <img src="results/AntBulletEnv-v0/baseline.png" alt="AntBulletEnv" width="500px">

## Reference

John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and Pieter Abbeel. High-dimensional
continuous control using generalized advantage estimation. CoRR, abs/1506.02438, 2015.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy
optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail


