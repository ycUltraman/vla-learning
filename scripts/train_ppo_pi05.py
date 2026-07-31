"""Phase 2: PI0.5-based PPO for Panda pick-and-place.

Loads BC checkpoint, wraps PI0.5 with stochastic action + value head,
runs PPO in MuJoCo. ~500ms per step due to PI0.5 inference.

Usage (on server):
    source ~/autodl-tmp/venv_lerobot/bin/activate
    export HF_HOME=/root/autodl-tmp/.hf_cache HF_HUB_OFFLINE=1
    python train_ppo_pi05.py --checkpoint <path> --episodes 100
"""

import argparse, sys, time, warnings
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.distributions import Normal

from panda_rl_env import PandaRLEnv


# ── PI0.5 Wrapper with stochastic head + value head ────────────

class PI05Stochastic(nn.Module):
    """Wrap frozen PI0.5 with learnable action log_std and value head."""

    def __init__(self, pi05_policy, preprocessor, postprocessor, device,
                 res_scale=0.01, hidden_dim=1024):
        super().__init__()
        self.pi05 = pi05_policy  # frozen
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.residual_scale = res_scale  # configurable

        # Residual head: bc_action + ee + cube + rel + dist + grip → correction
        # 7 + 3 + 3 + 3 + 1 + 1 = 18
        self.residual = nn.Sequential(
            nn.Linear(18, 64), nn.Tanh(),
            nn.Linear(64, 7),
        ).to(device)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

        # Learnable log_std for residual noise (very small)
        self.log_std = nn.Parameter(torch.full((7,), -7.0, device=device))  # std ≈ 0.001, near-deterministic

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.ReLU(),
            nn.Linear(512, 1),
        ).to(device)

        # Register hook to capture action head input
        self._action_features = None
        self._hook_handle = None
        self._register_hook()

    def _register_hook(self):
        """Hook into PI0.5 to capture features before action head."""
        # Find the action_in_proj or equivalent layer
        for name, module in self.pi05.named_modules():
            if name.endswith("action_in_proj"):
                self._hook_handle = module.register_forward_hook(self._hook_fn)
                break
        if self._hook_handle is None:
            print("WARNING: action_in_proj not found, using zero features for value")

    def _hook_fn(self, module, input, output):
        self._action_features = output.detach()  # (batch, seq, dim)

    def forward(self, raw_batch, xy_dist=None, cube_xyz=None):
        """Run PI0.5 → bc_action, then add learnable residual correction.

        xy_dist: EE-cube distance in meters.
        cube_xyz: cube position in world frame (3,). Used as residual context.
        """
        batch = self.preprocessor(raw_batch)

        # Get BC action (deterministic, frozen)
        with torch.no_grad():
            raw_action = self.pi05.select_action(batch)
            bc_action = self.postprocessor(raw_action).squeeze(0).to(self.device).clone()

        # Build normalized residual input
        # bc_action ~[-0.02,0.02], ee ~[-1,1], cube ~[-0.5,0.5], dist ~[0,1], grip ~[0,0.08]
        # Normalize to ~[-1,1] so bc_action isn't dwarfed by position values
        state = raw_batch["observation.state"][0]
        ee_xyz = state[8:11] / 1.0              # already ~[-1,1]
        grip = state[7:8] / 0.08                # normalize to ~[0,1]
        dist_t = torch.tensor([xy_dist if xy_dist is not None else 0.0],
                              device=self.device, dtype=torch.float32) / 0.5  # ~[0,1]
        if cube_xyz is not None:
            cube_t = torch.tensor(cube_xyz, device=self.device, dtype=torch.float32) / 0.5
        else:
            cube_t = torch.zeros(3, device=self.device)
        bc_norm = bc_action.detach() / 0.02     # scale to ~[-1,1]
        # Add relative position: cube - ee (tells network which direction to move)
        rel_xyz = cube_t - ee_xyz
        res_input = torch.cat([bc_norm, ee_xyz, cube_t, rel_xyz, dist_t, grip])
        delta = self.residual(res_input)
        delta = torch.clamp(delta, -self.residual_scale, self.residual_scale)

        # Gating uses 3D pose distance — XY may be close but Z still 40cm high
        if xy_dist is not None:
            # Compute full 3D pose distance
            z_init = cube_t[2].item() * 0.5 + 0.04  # un-normalize: cube_z*0.5 + offset
            ee_z = ee_xyz[2].item()
            z_err = abs(ee_z - z_init)
            pose_dist = float(np.sqrt(xy_dist**2 + z_err**2))
            if pose_dist > 0.25:   w = 0.0    # far: pure BC
            elif pose_dist > 0.12: w = 0.2    # approaching
            elif pose_dist > 0.05: w = 0.7    # close
            else:                 w = 1.0     # grasp zone
            delta_xyz = delta[:3] * w
        else:
            delta_xyz = delta[:3]

        # Tight Z clamp: XY matters more than Z for grasping
        delta_z = torch.clamp(delta[2:3], -0.001, 0.003)
        delta = torch.cat([delta_xyz[:2], delta_z, delta[3:]], dim=0)

        mean = bc_action + delta

        # Small learnable noise
        std = torch.exp(self.log_std.clamp(-5, -2))

        # Value
        if self._action_features is not None:
            feat = self._action_features.mean(dim=1).squeeze(0)
        else:
            feat = torch.zeros(1024, device=self.device)
        value = self.value_head(feat.float()).squeeze(-1)

        return mean, std, value, bc_action  # bc_action = pure BC output

    def act(self, batch, xy_dist=None, cube_xyz=None):
        """Sample action. Returns (action, log_prob, value, ppo_mean, bc_mean)."""
        mean, std, value, bc_action = self.forward(batch, xy_dist=xy_dist, cube_xyz=cube_xyz)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action, log_prob, value, mean, bc_action

    def trainable_parameters(self):
        """Return trainable parameters (log_std + value head)."""
        params = list(self.value_head.parameters()) + [self.log_std]
        return params


# ── Preprocessing (matches PI0.5 training) ─────────────────────

def build_batch(obs_dict, device):
    """Convert env observation dict → raw PI0.5 batch dict (preprocessor handles conversion)."""
    import torch

    front = torch.from_numpy(obs_dict["observation.images.front"]).float() / 255.0
    wrist = torch.from_numpy(obs_dict["observation.images.wrist"]).float() / 255.0
    state = torch.from_numpy(obs_dict["observation.state"]).float()

    # Preprocessor expects CHW float [0,1] tensors
    return {
        "observation.images.front": front.permute(2, 0, 1).unsqueeze(0).to(device),
        "observation.images.wrist": wrist.permute(2, 0, 1).unsqueeze(0).to(device),
        "observation.state": state.unsqueeze(0).to(device),
        "task": "move to the red cube and pick it up",
    }


# ── PPO Buffer ──────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def add(self, obs_dict, action, reward, log_prob, value, bc_mean, done):
        self.obs.append(obs_dict)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.bc_means.append(bc_mean)
        self.dones.append(done)

    def clear(self):
        self.obs = []; self.actions = []; self.rewards = []
        self.log_probs = []; self.values = []; self.bc_means = []
        self.dones = []

    def size(self):
        return len(self.actions)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_val = 0.0 if dones[t] else values[t + 1]
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    return advantages, advantages + values


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to PI0.5 BC checkpoint dir (pretrained_model/)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--save", default="./pi05_ppo.pt")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bc_loss", type=float, default=5, help="BC regularization weight")
    parser.add_argument("--res_scale", type=float, default=0.01, help="Max residual correction (m)")
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Device: {device}")

    # ── Load PI0.5 ──
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.configs import PreTrainedConfig
    from peft import PeftConfig, PeftModel

    print(f"Loading PI0.5 from {args.checkpoint} ...")
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.pretrained_path = args.checkpoint
    peft_config = PeftConfig.from_pretrained(args.checkpoint)
    base_policy = PI05Policy.from_pretrained(
        peft_config.base_model_name_or_path, config=cfg)
    pi05 = PeftModel.from_pretrained(base_policy, args.checkpoint, config=peft_config)
    pi05 = pi05.merge_and_unload()
    pi05 = pi05.to(device=device, dtype=torch.float32)

    # Selective freeze: vision + LLM + projector frozen, only action head trainable
    trainable_keys = ['action', 'state_proj']
    n_trainable = 0
    for name, p in pi05.named_parameters():
        if any(k in name for k in trainable_keys):
            p.requires_grad = True
            n_trainable += p.numel()
        else:
            p.requires_grad = False
    print(f"PI0.5 trainable: {n_trainable/1e6:.1f}M params (action head only)")

    # Load preprocessor + postprocessor (handles resize, normalize, unnormalize)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=pi05.config, pretrained_path=args.checkpoint)
    print("PI0.5 loaded and frozen.")

    # ── Wrap with stochastic head + value ──
    policy = PI05Stochastic(pi05, preprocessor, postprocessor, device, res_scale=args.res_scale)
    # pi05 is stored as submodule, so policy.parameters() includes everything
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad], lr=args.lr)

    # ── Env ──
    env = PandaRLEnv()
    buffer = RolloutBuffer()
    reward_history = []

    print(f"Starting PPO: {args.episodes} episodes")
    t_start = time.time()
    success_count = 0

    for ep in range(args.episodes):
        obs_dict = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
        obs_dict = env.get_obs_pi05()
        done = False
        ep_reward = 0.0
        ep_start = time.time()
        ep_actions = []
        ep_residuals = []  # |ppo_mean - bc|
        ep_bc = []         # pure BC action
        ep_ppo = []        # PPO mean (bc + residual)

        while not done:
            batch = build_batch(obs_dict, device)
            # Compute EE-cube distance + cube position for residual context
            ee_xy = obs_dict["observation.state"][8:11][:2]
            cube_pos = env.cube_position.copy()
            xy_dist = float(np.linalg.norm(ee_xy - cube_pos[:2]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                action, log_prob, value, ppo_mean, bc_mean = policy.act(
                    batch, xy_dist=xy_dist, cube_xyz=cube_pos)

            action_np = action.detach().cpu().numpy()
            ppo_np = ppo_mean.detach().cpu().numpy()
            bc_np = bc_mean.detach().cpu().numpy()
            ep_residuals.append(np.abs(ppo_np - bc_np))
            ep_bc.append(bc_np.copy())
            ep_ppo.append(ppo_np.copy())
            # Clip EE delta to safe range (±0.05m per step)
            action_np[:3] = np.clip(action_np[:3], -0.05, 0.05)
            ep_actions.append(action_np.copy())
            # PI0.5 outputs 7D [dx,dy,dz,drx,dry,drz,grip]
            # RL env expects 4D [dx,dy,dz,grip]
            rl_action = np.array([
                action_np[0], action_np[1], action_np[2], action_np[6]
            ])

            next_obs_14, reward, terminated, truncated, _ = env.step(rl_action)
            done = terminated or truncated
            ep_reward += reward

            buffer.add(obs_dict, action.detach(),
                       reward, log_prob.detach(), value.detach(),
                       bc_mean.detach(), done)  # bc_mean = pure frozen BC output

            obs_dict = env.get_obs_pi05()

        # PPO update
        if buffer.size() >= 4:
            n = buffer.size()
            rewards_t = torch.tensor(buffer.rewards, dtype=torch.float32, device=device)
            values_t = torch.stack(buffer.values)
            dones_t = torch.tensor(buffer.dones, dtype=torch.float32, device=device)
            log_probs_t = torch.stack(buffer.log_probs)
            actions_t = torch.stack(buffer.actions)
            bc_means_t = torch.stack(buffer.bc_means) if buffer.bc_means else None

            adv_t, ret_t = compute_gae(rewards_t, values_t, dones_t)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            # Debug: print advantage stats every 5 episodes
            if True:  # print every episode
                print(f"  [debug] adv mean={adv_t.mean():+.4f} std={adv_t.std():.4f} "
                      f"ret mean={ret_t.mean():+.2f}")

            for _ in range(5):
                for t in range(n):
                    batch_t = build_batch(buffer.obs[t], device)
                    mean_t, std_t, _, _ = policy.forward(batch_t)
                    dist = Normal(mean_t, std_t)
                    new_lp = dist.log_prob(actions_t[t]).sum()
                    old_lp = log_probs_t[t]
                    ratio = torch.exp(new_lp - old_lp)
                    surr1 = ratio * adv_t[t]
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[t]
                    policy_loss = -torch.min(surr1, surr2)

                    # BC regularization: protect BC from harmful deviation
                    bc_loss = args.bc_loss * ((mean_t - bc_means_t[t]) ** 2).sum()

                    # Value loss
                    _, _, val_t, _ = policy.forward(batch_t)
                    value_loss = nn.functional.mse_loss(val_t, ret_t[t])

                    loss = policy_loss + bc_loss + 0.5 * value_loss
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in policy.parameters() if p.requires_grad], 1.0)
                    optimizer.step()

        buffer.clear()
        reward_history.append(ep_reward)

        dt = time.time() - ep_start
        if env._success:
            success_count += 1
        if True:  # print every episode
            avg_r = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else np.mean(reward_history)
            print(f"Ep {ep:3d} | r: {ep_reward:+.1f} | avg10: {avg_r:+.1f} | "
                  f"steps: {env._step_count} | {dt:.0f}s | success: {env._success} | total_success: {success_count}/{ep+1}")

        if True:  # print every episode
            resids = np.array(ep_residuals)
            # 1 line: residual magnitude (dx,dy,dz,grip only — key diagnostic)
            print(f"  residual dx={resids[:,0].mean():.4f} dy={resids[:,1].mean():.4f} dz={resids[:,2].mean():.4f} grip={resids[:,6].mean():.4f}")
            # 1 line: BC vs PPO (mean, no noise) at step 0
            if ep_bc and ep_ppo:
                bc0, ppo0 = ep_bc[0], ep_ppo[0]
                print(f"  BC=({bc0[0]:+.4f},{bc0[1]:+.4f},{bc0[2]:+.4f}) PPO=({ppo0[0]:+.4f},{ppo0[1]:+.4f},{ppo0[2]:+.4f})")
            # 1 line: trajectory final state (init cube = fixed target, current = may be pushed)
            cube_init = getattr(env, '_init_cube', env.cube_position)
            cube_cur, ee = env.cube_position, env.ee_position
            print(f"  cube_init=({cube_init[0]:.2f},{cube_init[1]:.2f},{cube_init[2]:.3f}) cube_now=({cube_cur[0]:.2f},{cube_cur[1]:.2f},{cube_cur[2]:.3f}) ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) xy_to_init={np.linalg.norm(ee[:2]-cube_init[:2]):.3f} grasped={env._grasped}")
            print(f"  rew: prog={env._rew_progress:+.1f} pose={env._rew_pose:+.1f} grasp={env._rew_grasp:+.1f} succ={env._rew_success:+.1f} att={env._rew_attempt:+.1f} step={env._rew_step:+.1f} term={env._rew_terminal:+.1f}")

    total_time = time.time() - t_start
    print(f"\nDone. {args.episodes} episodes in {total_time/60:.0f}min")
    torch.save({
        "log_std": policy.log_std.data,
        "value_head": policy.value_head.state_dict(),
    }, args.save)
    env.close()


if __name__ == "__main__":
    main()
