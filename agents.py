import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy
from torch.cuda.amp import autocast

from sub_models.functions_losses import SymLogTwoHotLoss
from utils import EMAScalar


def percentile(x, percentage):
    flat_x = torch.flatten(x)
    kth = int(percentage*len(flat_x))
    per = torch.kthvalue(flat_x, kth).values
    return per


def calc_lambda_return(rewards, values, termination, gamma, lam, dtype=torch.float32, device="cuda"):
    # Invert termination to have 0 if the episode ended and 1 otherwise
    inv_termination = (termination * -1) + 1

    batch_size, batch_length = rewards.shape[:2]
    # gae_step = torch.zeros((batch_size, ), dtype=dtype, device="cuda")
    gamma_return = torch.zeros((batch_size, batch_length+1), dtype=dtype, device=device)
    gamma_return[:, -1] = values[:, -1]
    for t in reversed(range(batch_length)):  # with last bootstrap
        gamma_return[:, t] = \
            rewards[:, t] + \
            gamma * inv_termination[:, t] * (1-lam) * values[:, t] + \
            gamma * inv_termination[:, t] * lam * gamma_return[:, t+1]
    return gamma_return[:, :-1]


class ActorCriticAgent(nn.Module):
    def __init__(self, feat_dim, num_layers, hidden_dim, action_dim, gamma, lambd, entropy_coef,
                 action_kind="discrete", min_std=0.1, max_std=1.0, clip_sample=False,
                 policy_init_scale=0.01) -> None:
        super().__init__()
        self.gamma = gamma
        self.lambd = lambd
        self.entropy_coef = entropy_coef
        # "discrete": a Categorical over action_dim actions (STORM). "continuous": a
        # DreamerV3-style bounded normal over an action_dim vector in [-1, 1] -- mean
        # tanh(mu), std in [min_std, max_std] -- trained with the same REINFORCE loss.
        if action_kind not in ("discrete", "continuous"):
            raise ValueError(f"Unknown action_kind: {action_kind!r}")
        self.action_kind = action_kind
        self.action_dim = action_dim
        self.min_std = min_std
        self.max_std = max_std
        # Continuous only. The policy is scored on its RAW sample, as DreamerV3 does: the
        # [-1, 1] clip happens where the action is executed (sample_as_env_action, and the
        # world model's input in encode_action), so it is part of the environment and the
        # REINFORCE gradient stays unbiased. clip_sample=True restores the earlier
        # behaviour -- scoring the clipped action with the Gaussian density -- which
        # biases the std gradient towards the bounds (Fujita & Maeda 2018, "Clipped
        # Action Policy Gradient").
        self.clip_sample = clip_sample
        self.use_amp = True
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32

        self.symlog_twohot_loss = SymLogTwoHotLoss(255, -20, 20)

        actor = [
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ]
        for i in range(num_layers - 1):
            actor.extend([
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
        self.actor = nn.Sequential(
            *actor,
            nn.Linear(hidden_dim, action_dim if action_kind == "discrete" else 2 * action_dim)
        )
        if action_kind == "continuous" and policy_init_scale != 1.0:
            # DreamerV3's policy outscale: start near mean 0 / std ~0.9, away from where
            # tanh (mean) and sigmoid (std) saturate. 1.0 = PyTorch's default init.
            with torch.no_grad():
                self.actor[-1].weight.mul_(policy_init_scale)
                self.actor[-1].bias.zero_()

        critic = [
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ]
        for i in range(num_layers - 1):
            critic.extend([
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])

        self.critic = nn.Sequential(
            *critic,
            nn.Linear(hidden_dim, 255)
        )
        self.slow_critic = copy.deepcopy(self.critic)

        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-5, eps=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @torch.no_grad()
    def update_slow_critic(self, decay=0.98):
        for slow_param, param in zip(self.slow_critic.parameters(), self.critic.parameters()):
            slow_param.data.copy_(slow_param.data * decay + param.data * (1 - decay))

    def policy(self, x):
        logits = self.actor(x)
        return logits

    def action_dist(self, logits):
        """The policy distribution for actor outputs ``logits``."""
        if self.action_kind == "discrete":
            return distributions.Categorical(logits=logits)
        mean, std = logits.float().chunk(2, dim=-1)
        mean = torch.tanh(mean)
        std = (self.max_std - self.min_std) * torch.sigmoid(std + 2.0) + self.min_std
        return distributions.Independent(distributions.Normal(mean, std), 1)

    def value(self, x):
        value = self.critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    @torch.no_grad()
    def slow_value(self, x):
        value = self.slow_critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    def get_logits_raw_value(self, x):
        logits = self.actor(x)
        raw_value = self.critic(x)
        return logits, raw_value

    @torch.no_grad()
    def sample(self, latent, greedy=False):
        self.eval()
        with torch.autocast(device_type=latent.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            logits = self.policy(latent)
            if self.action_kind == "discrete":
                dist = distributions.Categorical(logits=logits)
                if greedy:
                    action = dist.probs.argmax(dim=-1)
                else:
                    action = dist.sample()
            else:
                dist = self.action_dist(logits)
                action = dist.mean if greedy else dist.sample()
                if self.clip_sample:
                    action = action.clamp(-1.0, 1.0)
        return action

    def sample_as_env_action(self, latent, greedy=False):
        action = self.sample(latent, greedy)
        if self.action_kind == "discrete":
            return action.detach().cpu().squeeze(-1).numpy()
        # (B, 1, A) -> (B, A) float32 for the vector env, clipped to its bounds: the env
        # executes (and the replay buffer stores) the clipped action.
        return action.detach().float().clamp(-1.0, 1.0).cpu().squeeze(1).numpy()

    def update(self, latent, action, old_logprob, old_value, reward, termination, logger=None, device="cuda",
               step=None):
        '''
        Update policy and value model
        '''
        self.train()
        with torch.autocast(device_type=latent.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            logits, raw_value = self.get_logits_raw_value(latent)
            dist = self.action_dist(logits[:, :-1])
            log_prob = dist.log_prob(action if self.action_kind == "discrete" else action.float())
            entropy = dist.entropy()

            # decode value, calc lambda return
            slow_value = self.slow_value(latent)
            slow_lambda_return = calc_lambda_return(reward, slow_value, termination, self.gamma, self.lambd, device=device)
            value = self.symlog_twohot_loss.decode(raw_value)
            lambda_return = calc_lambda_return(reward, value, termination, self.gamma, self.lambd, device=device)

            # update value function with slow critic regularization
            value_loss = self.symlog_twohot_loss(raw_value[:, :-1], lambda_return.detach())
            slow_value_regularization_loss = self.symlog_twohot_loss(raw_value[:, :-1], slow_lambda_return.detach())

            lower_bound = self.lowerbound_ema(percentile(lambda_return, 0.05))
            upper_bound = self.upperbound_ema(percentile(lambda_return, 0.95))
            S = upper_bound-lower_bound
            norm_ratio = torch.max(torch.ones(1).to(device), S)  # max(1, S) in the paper
            norm_advantage = (lambda_return-value[:, :-1]) / norm_ratio
            policy_loss = -(log_prob * norm_advantage.detach()).mean()

            entropy_loss = entropy.mean()

            loss = policy_loss + value_loss + slow_value_regularization_loss - self.entropy_coef * entropy_loss

        # gradient descent
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.update_slow_critic()

        if logger is not None:
            logger.log('ActorCritic/policy_loss', policy_loss.item(), step=step)
            logger.log('ActorCritic/value_loss', value_loss.item(), step=step)
            logger.log('ActorCritic/entropy_loss', entropy_loss.item(), step=step)
            logger.log('ActorCritic/S', S.item(), step=step)
            logger.log('ActorCritic/norm_ratio', norm_ratio.item(), step=step)
            logger.log('ActorCritic/total_loss', loss.item(), step=step)
            # Imagined-return diagnostics: what the policy *thinks* it earns in the
            # dream, to compare against real-env return (sample/* and eval/*). High
            # imagined return with low real return = objective mismatch: the agent
            # is exploiting an inaccurate world model rather than solving the task.
            logger.log('ActorCritic/imagined_return', lambda_return.mean().item(), step=step)
            logger.log('ActorCritic/imagined_reward_mean', reward.mean().item(), step=step)
            logger.log('ActorCritic/imagined_termination_frac', termination.float().mean().item(), step=step)
            if self.action_kind == "continuous":
                # Is the policy pinned at a bound? std at its floor/ceiling, how saturated
                # tanh(mean) is, and how often a raw sample falls outside [-1, 1].
                with torch.no_grad():
                    std = dist.base_dist.stddev
                    band = 0.01 * (self.max_std - self.min_std)
                    logger.log('ActorCritic/policy_std_mean', std.mean().item(), step=step)
                    logger.log('ActorCritic/policy_std_at_floor', (std < self.min_std + band).float().mean().item(), step=step)
                    logger.log('ActorCritic/policy_std_at_ceiling', (std > self.max_std - band).float().mean().item(), step=step)
                    logger.log('ActorCritic/policy_abs_mean', dist.base_dist.mean.abs().mean().item(), step=step)
                    logger.log('ActorCritic/action_clip_frac', (action.float().abs() > 1).float().mean().item(), step=step)
