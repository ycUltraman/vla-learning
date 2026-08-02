"""Value-only PPO: learn quality scores for BC trajectories, no action modification.

Rollout: pure BC (proven 50%). PPO updates only the value head.
The value function learns to predict which episodes succeed vs fail.
"""

import argparse, sys, time, warnings
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
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


class ValueHead(nn.Module):
    """Small value network on top of state features."""
    def __init__(self, device):
        super().__init__()
        # State: 15D + extra features from env
        self.net = nn.Sequential(
            nn.Linear(15, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)

    def forward(self, x):
        return self.net(x)


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.rewards = []
        self.dones = []

    def add(self, state_feat, reward, done):
        self.states.append(state_feat)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.rewards.clear()
        self.dones.clear()

    def size(self):
        return len(self.rewards)


def compute_returns(rewards, dones, gamma=0.99):
    """Compute discounted returns."""
    returns = []
    R = 0.0
    for r, d in zip(reversed(rewards), reversed(dones)):
        R = r + gamma * R * (1 - d)
        returns.append(R)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--save", default="./value_head.pt")
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Device: {device}")

    # Load PI0.5 (same as before)
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.configs import PreTrainedConfig
    from peft import PeftConfig, PeftModel

    print(f"Loading PI0.5 from {args.checkpoint} ...")
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.pretrained_path = args.checkpoint
    peft_config = PeftConfig.from_pretrained(args.checkpoint)
    base_policy = PI05Policy.from_pretrained(peft_config.base_model_name_or_path, config=cfg)
    pi05 = PeftModel.from_pretrained(base_policy, args.checkpoint, config=peft_config)
    pi05 = pi05.merge_and_unload()
    pi05 = pi05.to(device=device, dtype=torch.float32)
    pi05.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=pi05.config, pretrained_path=args.checkpoint)
    print("PI0.5 loaded.")

    env = PandaRLEnv()
    value_net = ValueHead(device)
    optimizer = torch.optim.Adam(value_net.parameters(), lr=args.lr)
    buffer = RolloutBuffer()
    success_history = []

    for ep in range(args.episodes):
        env.reset()
        obs = env.get_obs_pi05()
        done = False
        ep_reward = 0.0

        while not done:
            # Get BC action (no PPO modification)
            proc_batch = preprocessor(build_batch(obs, device))
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = pi05.select_action(proc_batch)
                bc_action = postprocessor(raw).squeeze(0).cpu().numpy()

            rl_action = np.array([bc_action[0], bc_action[1], bc_action[2], bc_action[6]])

            # State features: concatenate state + EE + cube + dist
            ee = env.ee_position
            cube = env.cube_position
            state_feat = np.concatenate([
                obs["observation.state"][:11],  # joints + grip + ee
                cube,                            # cube position
                [np.linalg.norm(ee - cube)],     # distance
            ]).astype(np.float32)

            next_obs_14, reward, terminated, truncated, _ = env.step(rl_action)
            done = terminated or truncated
            ep_reward += reward

            buffer.add(state_feat, reward, done)
            obs = env.get_obs_pi05()

        # Value update
        if buffer.size() > 0:
            returns = compute_returns(buffer.rewards, buffer.dones).to(device)
            states_t = torch.tensor(np.array(buffer.states), dtype=torch.float32, device=device)

            for _ in range(10):
                pred = value_net(states_t).squeeze(-1)
                loss = nn.functional.mse_loss(pred, returns)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
                optimizer.step()

        buffer.clear()
        if env._success:
            success_history.append(1)
        else:
            success_history.append(0)

        if ep % 20 == 0 or ep < 10:
            recent = np.mean(success_history[-20:]) if len(success_history) >= 20 else np.mean(success_history)
            print(f"Ep {ep:4d} | r: {ep_reward:+.1f} | "
                  f"steps: {env._step_count} | succ_rate: {recent*100:.0f}% | "
                  f"total_succ: {sum(success_history)}/{ep+1}")

    final_rate = np.mean(success_history[-100:]) if len(success_history) >= 100 else np.mean(success_history)
    print(f"\nFinal success rate (last 100): {final_rate*100:.0f}%")
    torch.save({"value_head": value_net.state_dict()}, args.save)
    env.close()


if __name__ == "__main__":
    main()
