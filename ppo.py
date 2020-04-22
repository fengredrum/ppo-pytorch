import torch
import torch.nn as nn
import torch.optim as optim


class PPO:
    def __init__(self,
                 actor_critic,
                 device,
                 lr=3e-4,
                 eps=1e-5,
                 max_grad_norm=None,
                 clip_param=0.2,
                 dual_clip_param=5.,
                 ppo_epoch=10,
                 num_mini_batch=32,
                 value_loss_coef=0.5,
                 entropy_coef=0.01,
                 norm_advantage=False,
                 clip_value_loss=True,
                 ):

        # NN policy
        self.actor_critic = actor_critic

        # PPO config
        self.clip_param = clip_param
        if dual_clip_param:
            self.dual_clip_param = torch.tensor(dual_clip_param,
                                                dtype=torch.float32, device=device)
        else:
            self.dual_clip_param = dual_clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.norm_advantage = norm_advantage

        # Loss config
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.clip_value_loss = clip_value_loss

        # Optimizer config
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.kl = nn.KLDivLoss()

    def update(self, buffer):
        advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        if self.norm_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        value_loss_mean = 0.
        action_loss_mean = 0.
        entropy_mean = 0.
        kl_mean = 0.

        for e in range(self.ppo_epoch):
            data_generator = buffer.batch_data_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_value_act(
                    obs_batch, actions_batch)

                # Compute action loss
                kl_loss = self.kl(action_log_probs, old_action_log_probs_batch.exp())

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr_1 = ratio * adv_targ
                surr_2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                     1.0 + self.clip_param) * adv_targ
                if self.dual_clip_param:
                    action_loss = -torch.max(
                        torch.min(surr_1, surr_2),
                        self.dual_clip_param * adv_targ).mean()
                else:
                    action_loss = -torch.min(surr_1, surr_2).mean()

                # Compute value loss
                if self.clip_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)

                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)

                    value_loss = 0.5 * torch.min(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # Update policy
                self.optimizer.zero_grad()
                (action_loss +
                 value_loss * self.value_loss_coef -
                 dist_entropy * self.entropy_coef).backward()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                             self.max_grad_norm)
                self.optimizer.step()

                value_loss_mean += value_loss.item()
                action_loss_mean += action_loss.item()
                entropy_mean += dist_entropy.item()
                kl_mean += kl_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_mean /= num_updates
        action_loss_mean /= num_updates
        entropy_mean /= num_updates
        kl_mean /= num_updates

        return dict(action_loss_mean=action_loss_mean,
                    value_loss_mean=value_loss_mean,
                    entropy_mean=entropy_mean,
                    kl_mean=kl_mean,
                    )
