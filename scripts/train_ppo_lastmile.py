"""Last-mile PPO: positional refinement in final 5cm.

PI0.5 handles everything until EE-cube < 5cm.
Then a small RL policy adds positional correction to bring EE exactly on target.
Does NOT modify grip, rotation, or full trajectory.

Usage:
    MUJOCO_GL=egl python train_ppo_lastmile.py \
        --checkpoint <path> --episodes 5000
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


class LastMilePolicy(nn.Module):
    """EE-cube offset + gripper state → positional correction (dx,dy,dz)."""
    def __init__(self):
        super().__init__()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = nn.Sequential(
            nn.Linear(7, 64), nn.ReLU(),  # ee(3)+cube(3)+grip(1)
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3),             # dx, dy, dz only
        ).to(dev)
        self.log_std = nn.Parameter(torch.full((3,), -2.0, device=dev))  # std≈0.14

    def forward(self, feat):
        mean = self.net(feat)
        std = torch.exp(self.log_std.clamp(-3, 0))
        return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--save", default="./lastmile_policy.pt")
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Device: {device}")

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
    policy = LastMilePolicy()
    value_net = nn.Sequential(nn.Linear(7, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()), lr=args.lr)
    success_count = 0

    for ep in range(args.episodes):
        env.reset()
        obs = env.get_obs_pi05()
        done = False
        ep_reward = 0.0
        t0 = time.time()
        ep_feats, ep_actions, ep_rewards = [], [], []
        ep_logps, ep_values, ep_dones = [], [], []

        while not done:
            batch = pre(build_batch(obs, device))
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = pi05.select_action(batch)
                bc = post(raw).squeeze(0).cpu().numpy()

            ee = env.ee_position
            cube = env.cube_position
            dist = float(np.linalg.norm(ee - cube))

            if dist < 0.05:
                # LAST MILE: PPO adds positional correction
                feat = np.concatenate([ee, cube, [env.gripper_width]]).astype(np.float32)
                ft = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
                mean, std = policy(ft)
                dist_ppo = Normal(mean, std)
                correction = dist_ppo.sample()
                log_prob = dist_ppo.log_prob(correction).sum()
                with torch.no_grad():
                    value = value_net(ft).squeeze(-1)
                # Apply correction to BC action
                bc[0:3] += correction.detach().cpu().numpy() * 0.01  # scale: max ~1cm
            else:
                log_prob = torch.tensor(0.0, device=device)
                value = torch.tensor(0.0, device=device)
                feat = np.zeros(7, dtype=np.float32)

            rl = np.array([bc[0], bc[1], bc[2], bc[6]])
            _, reward, terminated, truncated, _ = env.step(rl)
            done = terminated or truncated
            ep_reward += reward

            ep_feats.append(feat)
            ep_actions.append(correction.detach() if dist < 0.05 else torch.zeros(3, device=device))
            ep_rewards.append(reward)
            ep_logps.append(log_prob.detach() if dist < 0.05 else torch.zeros(3, device=device))
            ep_values.append(value.detach())
            ep_dones.append(done)
            obs = env.get_obs_pi05()

        # PPO update (only last-mile corrections)
        n = len(ep_rewards)
        if n > 4:
            # Only update on steps where PPO was active
            active = [i for i in range(n) if ep_actions[i].abs().sum() > 0 or ep_dones[i]]
            if len(active) > 2:
                # GAE on full trajectory for credit assignment
                rewards_t = torch.tensor(ep_rewards, dtype=torch.float32, device=device)
                values_t = torch.stack(ep_values)
                dones_t = torch.tensor(ep_dones, dtype=torch.float32, device=device)

                advantages = torch.zeros(n, device=device)
                gae = 0.0
                for t in reversed(range(n)):
                    next_val = 0.0 if ep_dones[t] else values_t[t + 1]
                    delta = rewards_t[t] + 0.99 * next_val - values_t[t]
                    gae = delta + 0.99 * 0.95 * (1 - ep_dones[t]) * gae
                    advantages[t] = gae
                returns = advantages + values_t

                # Only update active steps
                for i in active:
                    if ep_actions[i].abs().sum() == 0: continue
                    feat_t = torch.tensor(ep_feats[i], dtype=torch.float32, device=device).unsqueeze(0)
                    mean, std = policy(feat_t)
                    dist_ppo = Normal(mean, std)
                    new_lp = dist_ppo.log_prob(ep_actions[i].unsqueeze(0)).sum()
                    old_lp = ep_logps[i].sum()
                    ratio = torch.exp(new_lp - old_lp)
                    adv = advantages[i]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
                    policy_loss = -torch.min(surr1, surr2)

                    pred_v = value_net(feat_t).squeeze(-1)
                    value_loss = nn.functional.mse_loss(pred_v, returns[i])

                    loss = policy_loss + 0.5 * value_loss
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(policy.parameters()) + list(value_net.parameters()), 1.0)
                    optimizer.step()

        if env._success:
            success_count += 1

        if ep % 50 == 0 or ep < 10:
            print(f"Ep {ep:4d} | r: {ep_reward:+.1f} | {env._step_count}s | "
                  f"{time.time()-t0:.0f}s | succ: {success_count}/{ep+1}")

    print(f"\nFinal: {success_count}/{args.episodes} = {success_count/args.episodes*100:.0f}%")
    torch.save({"policy": policy.state_dict(),
                "value_net": value_net.state_dict()}, args.save)
    env.close()


if __name__ == "__main__":
    main()
