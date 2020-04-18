# ppo-pytorch

## Experiment Results

It takes about half an hour for $5e6$ training steps in a six cores MacBook Pro.

<p align="center">
  <img src="results/HalfCheetahBulletEnv-v0/baseline.png" alt="HalfCheetahBulletEnv"/>
</p>

<p align="center">
  <img src="results/AntBulletEnv-v0/baseline.png" alt="AntBulletEnv"/>
</p>

## Reference

John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and Pieter Abbeel. High-dimensional
continuous control using generalized advantage estimation. CoRR, abs/1506.02438, 2015.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy
optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
