"""Phase 1: State-only PPO for Franka Panda pick-and-place.

Uses a small MLP actor-critic (no vision) to verify the RL pipeline
before scaling to PI0.5.  ~1ms per step in MuJoCo.

Usage:
    python scripts/train_ppo.py --episodes 2000 --render
"""

import argparse
import sys
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).parent.parent))
from vla_inference.env.panda_rl_env import PandaRLEnv


# ── Actor-Critic Network ───────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=14, act_dim=4, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.shared(obs)
        mean = self.actor_mean(h)
        std = torch.exp(self.actor_log_std.clamp(-2, 0.5))
        value = self.critic(h)
        return mean, std, value

    def act(self, obs):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value, mean, std


# ── PPO Buffer ──────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def add(self, obs, action, reward, log_prob, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def get_batch(self):
        return (
            torch.stack(self.obs),
            torch.stack(self.actions),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.stack(self.log_probs),
            torch.stack(self.values),
            torch.tensor(self.dones, dtype=torch.float32),
        )


# ── GAE + PPO Update ────────────────────────────────────────────

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_val = 0.0 if dones[t] else values[t + 1].detach()
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def ppo_update(ac, optimizer, obs, actions, old_log_probs, advantages, returns,
               clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, epochs=10, batch_size=256):
    n = len(obs)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs):
        indices = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            b_obs = obs[idx]
            b_actions = actions[idx]
            b_old_lp = old_log_probs[idx]
            b_adv = advantages[idx]
            b_ret = returns[idx]

            mean, std, values = ac(b_obs)
            dist = Normal(mean, std)
            new_lp = dist.log_prob(b_actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratio = torch.exp(new_lp - b_old_lp)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values.squeeze(-1), b_ret)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
            optimizer.step()


# ── Training Loop ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save", type=str, default="./rl_checkpoint.pt")
    parser.add_argument("--bc_init", type=str, default=None,
                        help="Path to BC checkpoint for weight initialization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = PandaRLEnv()
    ac = ActorCritic(obs_dim=14, act_dim=4, hidden=256).to(device)

    # Load BC pretrained weights if provided
    if args.bc_init:
        bc = torch.load(args.bc_init, map_location=device)
        bc_state = bc["model_state"]
        ac.shared[0].weight.data.copy_(bc_state["net.0.weight"])
        ac.shared[0].bias.data.copy_(bc_state["net.0.bias"])
        ac.shared[2].weight.data.copy_(bc_state["net.2.weight"])
        ac.shared[2].bias.data.copy_(bc_state["net.2.bias"])
        ac.actor_mean.weight.data.copy_(bc_state["net.4.weight"])
        ac.actor_mean.bias.data.copy_(bc_state["net.4.bias"])
        print(f"Loaded BC init from {args.bc_init}")

    optimizer = torch.optim.Adam(ac.parameters(), lr=args.lr)
    buffer = RolloutBuffer()
    reward_history = deque(maxlen=50)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        done = False
        ep_reward = 0.0

        while not done:
            with torch.no_grad():
                action, log_prob, value, _, _ = ac.act(obs_t.unsqueeze(0))
            action_np = action.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            ep_reward += reward

            buffer.add(obs_t,
                       action.squeeze(0),
                       reward,
                       log_prob.squeeze(0),
                       value.squeeze(0),
                       done)

            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

        # PPO update after each episode
        obs_b, act_b, rew_b, old_lp_b, val_b, dones_b = buffer.get_batch()
        rew_b = rew_b.to(device); dones_b = dones_b.to(device)
        adv_b, ret_b = compute_gae(rew_b, val_b.squeeze(-1), dones_b)

        ppo_update(ac, optimizer, obs_b.to(device), act_b.to(device),
                   old_lp_b.to(device), adv_b.to(device), ret_b.to(device))

        buffer.clear()
        reward_history.append(ep_reward)

        if ep % 50 == 0:
            avg_r = np.mean(reward_history)
            print(f"Ep {ep:4d}/{args.episodes} | avg_reward: {avg_r:+.2f} | "
                  f"last_reward: {ep_reward:+.2f} | success: {env._success}")

        if (ep + 1) % 500 == 0:
            torch.save({"ac_state": ac.state_dict(), "opt_state": optimizer.state_dict()}, args.save)
            print(f"  -> saved {args.save}")

    torch.save({"ac_state": ac.state_dict(), "opt_state": optimizer.state_dict()}, args.save)
    print(f"Final model saved: {args.save}")
    env.close()


if __name__ == "__main__":
    main()
