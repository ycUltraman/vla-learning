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

        # Residual head: tiny random init → non-zero gradient for PPO
        # 7 + 3 + 3 + 3 + 1 + 1 = 18
        self.residual = nn.Sequential(
            nn.Linear(18, 64), nn.Tanh(),
            nn.Linear(64, 7),
        ).to(device)
        # Tiny init: enough for gradient flow, not enough to destroy BC
        nn.init.normal_(self.residual[-1].weight, std=0.001)
        nn.init.zeros_(self.residual[-1].bias)

        # Unified log_std — grip gets no extra noise (BC handles it)
        self.log_std_xyz = nn.Parameter(torch.full((6,), -7.0, device=device))
        self.log_std_grip = nn.Parameter(torch.tensor(-7.0, device=device))  # tight, BC knows grip
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

    def forward_train(self, raw_batch, bc_action_precomputed, xy_dist=None, cube_xyz=None):
        """PPO update version — uses precomputed BC action, no select_action call."""
        res_input = self._build_res_input(raw_batch, bc_action_precomputed, xy_dist, cube_xyz)
        delta = self.residual(res_input)
        delta = torch.clamp(delta, -self.residual_scale, self.residual_scale)

        if xy_dist is not None:
            if xy_dist > 0.30:   w = 0.0
            elif xy_dist > 0.15: w = 0.5
            elif xy_dist > 0.05: w = 0.8
            else:                w = 1.0
            delta_xyz = delta[:3] * w
        else:
            delta_xyz = delta[:3]

        delta_z = torch.clamp(delta[2:3], -0.001, 0.003)
        if xy_dist is not None and xy_dist < 0.06:
            grip_val = delta[6:7] * 2.0
        else:
            grip_val = torch.zeros_like(delta[6:7])
        delta = torch.cat([delta_xyz[:2], delta_z, delta[3:6], grip_val], dim=0)
        mean = bc_action_precomputed + delta

        std_xyz = torch.exp(self.log_std_xyz.clamp(-5, -2))
        std_grip = torch.exp(self.log_std_grip.clamp(-5, -2))
        std = torch.cat([std_xyz, std_grip.unsqueeze(0)])

        if self._action_features is not None:
            feat = self._action_features.mean(dim=1).squeeze(0)
        else:
            feat = torch.zeros(1024, device=self.device)
        value = self.value_head(feat.float()).squeeze(-1)
        return mean, std, value

    def _build_res_input(self, raw_batch, bc_action, xy_dist, cube_xyz):
        """Build normalized residual input without calling select_action."""
        state = raw_batch["observation.state"][0]
        ee_xyz = state[8:11] / 1.0
        grip = state[7:8] / 0.08
        dist_t = torch.tensor([xy_dist if xy_dist is not None else 0.0],
                              device=self.device, dtype=torch.float32) / 0.5
        if cube_xyz is not None:
            cube_t = torch.tensor(cube_xyz, device=self.device, dtype=torch.float32) / 0.5
            rel_xyz = cube_t - ee_xyz
        else:
            cube_t = torch.zeros(3, device=self.device)
            rel_xyz = torch.zeros(3, device=self.device)
        bc_norm = bc_action.detach() / 0.02
        return torch.cat([bc_norm, ee_xyz, cube_t, rel_xyz, dist_t, grip])

    def forward(self, raw_batch, xy_dist=None, cube_xyz=None):
        """Run PI0.5 → bc_action, then add learnable residual correction.

        xy_dist: EE-cube distance in meters.
        cube_xyz: cube position in world frame (3,). Used as residual context.
        """
        batch = self.preprocessor(raw_batch)

        # Get BC action
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
        # Grip: only adjust in grasp zone, and only slightly — BC knows how to grip
        if xy_dist is not None and xy_dist < 0.06:
            grip_val = delta[6:7] * 2.0  # ×2, small correction
        else:
            grip_val = torch.zeros_like(delta[6:7])  # far: pure BC
        delta = torch.cat([delta_xyz[:2], delta_z, delta[3:6], grip_val], dim=0)

        mean = bc_action + delta

        # Tight std for all dims — BC handles grip
        std_xyz = torch.exp(self.log_std_xyz.clamp(-5, -2))
        std_grip = torch.exp(self.log_std_grip.clamp(-5, -2))
        std = torch.cat([std_xyz, std_grip.unsqueeze(0)])

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

    # Add fresh LoRA on vision encoder for PPO
    # merge_and_unload bakes BC LoRA, then we add tiny trainable adapters
    from peft import LoraConfig, get_peft_model
    vis_pat = r".*vision_model\.encoder\.layers\.\d+\.self_attn\.(q_proj|v_proj|out_proj)"
    vis_lora = LoraConfig(r=8, lora_alpha=8, target_modules=vis_pat,
                          lora_dropout=0.0, bias="none",
                          task_type="FEATURE_EXTRACTION")
    pi05 = get_peft_model(pi05, vis_lora, adapter_name="ppo_vision")
    print("Vision LoRA added (r=4)")

    # Selective freeze: only action head + vision LoRA trainable
    trainable_keys = ['action', 'state_proj', 'lora_A', 'lora_B']
    n_trainable = 0
    for name, p in pi05.named_parameters():
        if any(k in name for k in trainable_keys):
            p.requires_grad = True
            n_trainable += p.numel()
        else:
            p.requires_grad = False
    n_lora = sum(1 for n,_ in pi05.named_parameters() if 'lora_' in n and n.endswith(('.weight','.bias')))
    print(f"PI0.5 trainable: {n_trainable/1e6:.1f}M params (action head + {n_lora} LoRA params)")

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
        # Clear PI0.5 action queue — PPO updates may have polluted it
        policy.pi05._action_queue.clear()
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
            # PPO policy = PI0.5 action head (gradient flows via select_action)
            proc_batch = preprocessor(build_batch(obs_dict, device))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = pi05.select_action(proc_batch)
            action_t = postprocessor(raw).squeeze(0).to(device)  # keep grad!

            # Add tiny noise for exploration
            std_t = torch.exp(policy.log_std_xyz.clamp(-5, -2))
            std_g = torch.exp(policy.log_std_grip.clamp(-5, -2))
            std_full = torch.cat([std_t, std_g.unsqueeze(0)])
            dist = torch.distributions.Normal(action_t, std_full)
            action_sample = dist.rsample()  # reparameterized sample
            log_prob_t = dist.log_prob(action_sample).sum()

            action_np = action_sample.detach().cpu().numpy()
            bc_np = action_t.detach().cpu().numpy()
            val_t = policy.value_head(torch.zeros(1024, device=device)).squeeze(-1)
            ep_residuals.append(np.zeros(7))
            ep_bc.append(bc_np)
            ep_ppo.append(bc_np)
            # Clip EE delta to safe range
            action_np[:3] = np.clip(action_np[:3], -0.05, 0.05)
            ep_actions.append(action_np.copy())
            # Map 7D → 4D for env
            rl_action = np.array([
                action_np[0], action_np[1], action_np[2], action_np[6]
            ])

            next_obs_14, reward, terminated, truncated, _ = env.step(rl_action)
            done = terminated or truncated
            ep_reward += reward

            # Store in buffer
            _act = torch.tensor(action_np, device=device, dtype=torch.float32)
            _lp = log_prob_t.detach() if torch.is_tensor(log_prob_t) else torch.tensor(0.0, device=device)
            _val = val_t.detach() if torch.is_tensor(val_t) else torch.tensor(0.0, device=device)
            _bc = torch.tensor(bc_np, device=device, dtype=torch.float32)
            buffer.add(obs_dict, _act.detach(), reward, _lp, _val, _bc.detach(), done)

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

            # Tiered replay: success=10, near-miss=3, failure=1
            if env._success:      n_epochs = 10
            elif env._grasped:    n_epochs = 3
            else:                 n_epochs = 1
            for _ in range(n_epochs):
                for t in range(n):
                    batch_t = build_batch(buffer.obs[t], device)
                    # Use forward_train with precomputed BC action — no select_action call
                    mean_t, std_t, val_t = policy.forward_train(
                        batch_t, bc_means_t[t], xy_dist=None, cube_xyz=None)
                    dist = Normal(mean_t, std_t)
                    new_lp = dist.log_prob(actions_t[t]).sum()
                    old_lp = log_probs_t[t]
                    ratio = torch.exp(new_lp - old_lp)
                    surr1 = ratio * adv_t[t]
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[t]
                    policy_loss = -torch.min(surr1, surr2)

                    # BC reg: 0 for first 10 eps (PPO explores), then ramp up
                    bc_weight = args.bc_loss * min(1.0, max(0.0, (ep - 10) / 10.0))
                    bc_loss = bc_weight * ((mean_t - bc_means_t[t]) ** 2).sum()

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
            # 1 line: BC vs PPO at step 0 (dx,dy,dz,grip)
            if ep_bc and ep_ppo:
                bc0, ppo0 = ep_bc[0], ep_ppo[0]
                print(f"  BC=({bc0[0]:+.4f},{bc0[1]:+.4f},{bc0[2]:+.4f},{bc0[6]:+.4f}) PPO=({ppo0[0]:+.4f},{ppo0[1]:+.4f},{ppo0[2]:+.4f},{ppo0[6]:+.4f})")
            # 1 line: trajectory final state (init cube = fixed target, current = may be pushed)
            cube_init = getattr(env, '_init_cube', env.cube_position)
            cube_cur, ee = env.cube_position, env.ee_position
            print(f"  cube_init=({cube_init[0]:.2f},{cube_init[1]:.2f},{cube_init[2]:.3f}) cube_now=({cube_cur[0]:.2f},{cube_cur[1]:.2f},{cube_cur[2]:.3f}) ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) xy_to_init={np.linalg.norm(ee[:2]-cube_init[:2]):.3f} grasped={env._grasped}")
            print(f"  rew: prog={env._rew_progress:+.1f} pose={env._rew_pose:+.1f} grasp={env._rew_grasp:+.1f} succ={env._rew_success:+.1f} att={env._rew_attempt:+.1f} step={env._rew_step:+.1f} term={env._rew_terminal:+.1f}")
            print(f"  min_pose_dist={getattr(env,'_min_pose_dist',99):.3f} ever_near={getattr(env,'_ever_near_grasp',False)}")
            with torch.no_grad():
                grip_std_val = torch.exp(policy.log_std_grip).item()
            print(f"  grip_std={grip_std_val:.4f} ppo_grip_mean={ep_ppo[0][6]:+.4f}")

    total_time = time.time() - t_start
    print(f"\nDone. {args.episodes} episodes in {total_time/60:.0f}min")
    torch.save({
        "log_std_xyz": policy.log_std_xyz.data,
        "log_std_grip": policy.log_std_grip.data,
        "value_head": policy.value_head.state_dict(),
    }, args.save)
    env.close()


if __name__ == "__main__":
    main()
