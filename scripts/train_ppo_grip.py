"""Grip-only PPO: train only gripper timing, leave XYZ unchanged.

Uses PI0.5 for XYZ (frozen), PPO trains a small grip policy.
The policy outputs a single grip value based on state features.
Proven: XYZ positioning is fine (min_pose_dist < 0.05), gripper timing fails.

Usage:
    MUJOCO_GL=egl python train_ppo_grip.py \
        --checkpoint <path> --episodes 200
"""

import argparse, warnings, time
import numpy as np
import torch, torch.nn as nn
from torch.distributions import Normal
from panda_rl_env import PandaRLEnv


def build_batch(obs_dict, device):
    front = torch.from_numpy(obs_dict["observation.images.front"]).float() / 255.0
    wrist = torch.from_numpy(obs_dict["observation.images.wrist"]).float() / 255.0
    state = torch.from_numpy(obs_dict["observation.state"]).float()
    return {
        "observation.images.front": front.permute(2, 0, 1).unsqueeze(0).to(device),
        "observation.images.wrist": wrist.permute(2, 0, 1).unsqueeze(0).to(device),
        "observation.state": state.unsqueeze(0).to(device),
        "task": "move to the red cube and pick it up",
    }


class GripPolicy(nn.Module):
    """State features → single grip value. Small, simple, trainable."""
    def __init__(self, feat_dim=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        self.log_std = nn.Parameter(torch.tensor(-1.0))  # std ≈ 0.37

    def forward(self, state_feat):
        mean = self.net(state_feat)  # (B, 1)
        std = torch.exp(self.log_std.clamp(-3, 0))
        return mean, std


class RolloutBuffer:
    def __init__(self):
        self.clear()
    def add(self, state_feat, grip_action, reward, log_prob, value, done):
        self.states.append(state_feat)
        self.actions.append(grip_action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.log_probs, self.values, self.dones = [], [], []


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_val = 0.0 if dones[t] else values[t + 1]
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    return advantages, advantages + values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--save", default="./grip_policy.pt")
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Device: {device}")

    # Load PI0.5 (frozen — does all XYZ positioning)
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.configs import PreTrainedConfig
    from peft import PeftConfig, PeftModel
    print(f"Loading PI0.5 from {args.checkpoint} ...")
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.pretrained_path = args.checkpoint
    peft_cfg = PeftConfig.from_pretrained(args.checkpoint)
    base = PI05Policy.from_pretrained(peft_cfg.base_model_name_or_path, config=cfg)
    pi05 = PeftModel.from_pretrained(base, args.checkpoint, config=peft_cfg)
    pi05 = pi05.merge_and_unload()
    pi05 = pi05.to(device=device, dtype=torch.float32)
    pi05.eval()
    pre, post = make_pre_post_processors(policy_cfg=pi05.config, pretrained_path=args.checkpoint)
    print("PI0.5 loaded.")

    env = PandaRLEnv()
    policy = GripPolicy(feat_dim=11).to(device)
    value_net = nn.Sequential(nn.Linear(11, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()), lr=args.lr)
    buffer = RolloutBuffer()
    success_count = 0

    for ep in range(args.episodes):
        env.reset()
        obs = env.get_obs_pi05()
        done = False
        ep_reward = 0.0
        t0 = time.time()

        while not done:
            # 1. Get BC action for XYZ (PI0.5 — frozen)
            batch = pre(build_batch(obs, device))
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = pi05.select_action(batch)
                bc = post(raw).squeeze(0).cpu().numpy()

            # 2. State features for grip policy
            state_feat = np.concatenate([
                obs["observation.state"][8:11],     # ee (3)
                env.cube_position,                    # cube (3)
                [obs["observation.state"][7]],        # grip width (1)
                [np.linalg.norm(env.ee_position[:2] - env.cube_position[:2])],  # xy dist
                [obs["observation.state"][8] - env.cube_position[0]],  # dx
                [obs["observation.state"][9] - env.cube_position[1]],  # dy
                [obs["observation.state"][10] - env.cube_position[2]],  # dz
            ]).astype(np.float32)

            # 3. Grip policy: output single grip value
            sf = torch.tensor(state_feat, dtype=torch.float32, device=device).unsqueeze(0)
            grip_mean, grip_std = policy(sf)
            dist = Normal(grip_mean, grip_std)
            grip_action = dist.sample()
            log_prob = dist.log_prob(grip_action).sum()

            # 4. Value
            with torch.no_grad():
                value = value_net(sf).squeeze(-1)

            # 5. Build full action: BC XYZ + PPO grip
            grip_val = grip_action.item()
            action_np = bc.copy()
            action_np[6] = np.clip(grip_val, -1.0, 1.0)  # replace grip
            rl_action = np.array([action_np[0], action_np[1], action_np[2], action_np[6]])

            _, reward, terminated, truncated, _ = env.step(rl_action)
            done = terminated or truncated
            ep_reward += reward

            buffer.add(sf.squeeze(0), grip_action.detach(), reward,
                       log_prob.detach(), value.detach(), done)
            obs = env.get_obs_pi05()

        # PPO update (grip only)
        n = len(buffer.states)
        if n > 4:
            rewards_t = torch.tensor(buffer.rewards, dtype=torch.float32, device=device)
            values_t = torch.stack(buffer.values)
            dones_t = torch.tensor(buffer.dones, dtype=torch.float32, device=device)
            logps_t = torch.stack(buffer.log_probs)
            acts_t = torch.stack(buffer.actions)
            states_t = torch.stack(buffer.states)

            adv_t, ret_t = compute_gae(rewards_t, values_t.squeeze(-1), dones_t)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            for _ in range(5):
                indices = torch.randperm(n)
                for i in range(0, n, min(128, n)):
                    idx = indices[i:i+128]
                    mean, std = policy(states_t[idx])
                    dist = Normal(mean, std)
                    new_lp = dist.log_prob(acts_t[idx]).sum(dim=-1)
                    old_lp = logps_t[idx]
                    ratio = torch.exp(new_lp - old_lp)
                    surr1 = ratio * adv_t[idx]
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[idx]
                    policy_loss = -torch.min(surr1, surr2).mean()

                    pred_v = value_net(states_t[idx]).squeeze(-1)
                    value_loss = nn.functional.mse_loss(pred_v, ret_t[idx])

                    loss = policy_loss + 0.5 * value_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(policy.parameters()) + list(value_net.parameters()), 1.0)
                    optimizer.step()

        buffer.clear()
        if env._success:
            success_count += 1

        dt = time.time() - t0
        if ep % 10 == 0:
            print(f"Ep {ep:4d} | r: {ep_reward:+.1f} | {env._step_count}s | "
                  f"{dt:.0f}s | succ: {success_count}/{ep+1}")

    print(f"\nFinal: {success_count}/{args.episodes} = {success_count/args.episodes*100:.0f}%")
    torch.save({"grip_policy": policy.state_dict(),
                "value_net": value_net.state_dict()}, args.save)
    env.close()


if __name__ == "__main__":
    main()
