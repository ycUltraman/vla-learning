"""Supervised residual policy: learn BC_action → optimal_action correction.

Collects BC trajectories, labels each step with the oracle correction
(target_xyz - bc_xyz), then trains a small MLP with MSE loss.

No reward, no GAE, no PI0.5 in training loop. Oracle = cube_position - ee_position.
"""

import argparse, warnings, time
import numpy as np
import mujoco
import torch, torch.nn as nn
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


class ResidualNet(nn.Module):
    """vis_proj(64) + state7 + bc_xyz(3) = 74D → Δxyz correction."""
    def __init__(self, vis_dim):
        super().__init__()
        self.vis_proj = nn.Linear(vis_dim, 64)
        self.net = nn.Sequential(
            nn.Linear(64 + 7 + 3, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3),
        )
        # Small init → initial correction ~0
        nn.init.normal_(self.net[-1].weight, std=0.001)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, vis_feat, state7, bc_xyz):
        v = self.vis_proj(vis_feat)
        x = torch.cat([v, state7, bc_xyz], dim=-1)
        raw = torch.tanh(self.net(x))
        # Per-axis output scales: xy ±2cm, z ±5mm
        scale = torch.tensor([0.02, 0.02, 0.005], device=x.device)
        return raw * scale


def collect_data(env, pi05, pre, post, vis_dim, device, episodes):
    """Run BC episodes, collect (vis_feat, state7, bc_xyz, delta_label) tuples."""
    _feat = None
    def _hook(m, inp, out): nonlocal _feat; _feat = out.detach()

    data = []  # list of (vis_feat, state7, bc_xyz, delta_label)
    successes = 0

    for ep in range(episodes):
        env.reset()
        obs = env.get_obs_pi05()
        done = False
        ep_data = []

        while not done:
            batch = pre(build_batch(obs, device)); _feat = None
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = pi05.select_action(batch)
                bc = post(raw).squeeze(0).cpu().numpy()

            vis_raw = _feat.squeeze(0) if _feat is not None else torch.zeros(vis_dim, device=device)
            if vis_raw.dim() == 2: vis_raw = vis_raw.mean(dim=0)
            vis_feat = vis_raw if vis_raw.dim() == 1 else vis_raw.squeeze(0)

            ee = env.ee_position
            cube = env.cube_position
            bc_xyz = bc[:3].copy()
            grip = env.gripper_width

            # Oracle: optimal step is toward the cube, clamped to reasonable magnitude
            raw_target = cube - ee
            target_xyz = np.clip(raw_target, -0.03, 0.03)
            delta_label = target_xyz - bc_xyz
            delta_label = np.clip(delta_label, -0.02, 0.02)

            ep_data.append((
                vis_feat.cpu().numpy() if isinstance(vis_feat, torch.Tensor) else vis_feat,
                np.concatenate([ee, cube, [grip]]).astype(np.float32),
                bc_xyz.astype(np.float32),
                delta_label.astype(np.float32),
            ))

            rl = np.array([bc[0], bc[1], bc[2], bc[6]])
            _, _, terminated, truncated, _ = env.step(rl)
            done = terminated or truncated
            obs = env.get_obs_pi05()

        if env._success:
            successes += 1

        n = len(ep_data)
        # Only keep episodes where we got reasonably close (some signal)
        min_dist = float('inf')
        for _, s, _, _ in ep_data:
            d = np.linalg.norm(s[3:6] - s[:3])  # cube - ee
            min_dist = min(min_dist, d)
        if min_dist < 0.10 or env._success:
            data.extend(ep_data)

        if ep % 10 == 0 or ep < 5:
            kept = "kept" if (min_dist < 0.10 or env._success) else "skip"
            print(f"  Collect ep {ep:3d} | steps={n} | min_dist={min_dist:.3f} | {kept} | succ={successes}")

    print(f"\nCollected {len(data)} samples from {episodes} episodes ({successes} successes)")
    return data


def train_residual(model, data, device, epochs=50, batch_size=256, lr=1e-3):
    """Train residual network on collected data."""
    # Prepare tensors
    vis_all = torch.tensor(np.stack([d[0] for d in data]), dtype=torch.float32).to(device)
    state_all = torch.tensor(np.stack([d[1] for d in data]), dtype=torch.float32).to(device)
    bc_all = torch.tensor(np.stack([d[2] for d in data]), dtype=torch.float32).to(device)
    label_all = torch.tensor(np.stack([d[3] for d in data]), dtype=torch.float32).to(device)

    n = len(data)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            pred = model(vis_all[idx], state_all[idx], bc_all[idx])
            loss = nn.functional.mse_loss(pred, label_all[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            batches += 1

        scheduler.step()
        if epoch % 10 == 0 or epoch < 5:
            avg = total_loss / max(batches, 1)
            print(f"  Epoch {epoch:3d} | loss={avg:.6f} | lr={scheduler.get_last_lr()[0]:.2e}")

    return model


def eval_residual(model, env, pi05, pre, post, vis_dim, device, episodes):
    """Evaluate residual policy: action = BC + residual."""
    _feat = None
    def _hook(m, inp, out): nonlocal _feat; _feat = out.detach()

    model.eval()
    successes = 0
    total_steps = 0

    for ep in range(episodes):
        env.reset()
        obs = env.get_obs_pi05()
        done = False

        while not done:
            batch = pre(build_batch(obs, device)); _feat = None
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = pi05.select_action(batch)
                bc = post(raw).squeeze(0).cpu().numpy()

            vis_raw = _feat.squeeze(0) if _feat is not None else torch.zeros(vis_dim, device=device)
            if vis_raw.dim() == 2: vis_raw = vis_raw.mean(dim=0)
            vis_feat = vis_raw if vis_raw.dim() == 1 else vis_raw.squeeze(0)

            ee = env.ee_position
            cube = env.cube_position
            state7 = torch.tensor(np.concatenate([ee, cube, [env.gripper_width]]),
                                  dtype=torch.float32, device=device).unsqueeze(0)
            bc_xyz = torch.tensor(bc[:3], dtype=torch.float32, device=device).unsqueeze(0)
            vis_t = vis_feat.unsqueeze(0) if vis_feat.dim() == 1 else vis_feat

            with torch.no_grad():
                delta = model(vis_t, state7, bc_xyz).squeeze(0).cpu().numpy()

            bc_c = bc.copy()
            bc_c[:3] = bc[:3] + delta
            rl = np.array([bc_c[0], bc_c[1], bc_c[2], bc[6]])
            _, _, terminated, truncated, _ = env.step(rl)
            done = terminated or truncated
            total_steps += 1
            obs = env.get_obs_pi05()

        if env._success:
            successes += 1
        if ep % 10 == 0 or ep < 5:
            print(f"  Eval ep {ep:3d} | steps={env._step_count} | succ={env._success}")

    rate = successes / episodes * 100
    print(f"\nResidual policy: {successes}/{episodes} = {rate:.1f}%")
    model.train()
    return rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--collect-episodes", type=int, default=200,
                        help="episodes for initial BC data collection")
    parser.add_argument("--train-epochs", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--dagger-iters", type=int, default=3,
                        help="DAgger iterations (collect→train→repeat)")
    parser.add_argument("--save", default="./residual_net.pt")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
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
    peft_cfg = PeftConfig.from_pretrained(args.checkpoint)
    base = PI05Policy.from_pretrained(peft_cfg.base_model_name_or_path, config=cfg)
    pi05 = PeftModel.from_pretrained(base, args.checkpoint, config=peft_cfg)
    pi05 = pi05.merge_and_unload()
    pi05 = pi05.to(device=device, dtype=torch.float32)
    pi05.eval()

    _feat = None
    def _hook(m, inp, out): nonlocal _feat; _feat = out.detach()
    vis_dim = 512
    for target in ["action_in_proj", "state_proj", "multi_modal_projector.linear"]:
        for n, m in pi05.named_modules():
            if n.endswith(target):
                m.register_forward_hook(_hook)
                if hasattr(m, 'out_features'): vis_dim = m.out_features
                print(f"  Hooked {n}, vis_dim={vis_dim}")
                break
        if vis_dim > 100:
            break

    pre, post = make_pre_post_processors(policy_cfg=pi05.config, pretrained_path=args.checkpoint)
    print("PI0.5 loaded.")

    # ── Env & Model ──
    env = PandaRLEnv()
    model = ResidualNet(vis_dim).to(device)
    print(f"ResidualNet: vis_dim={vis_dim}, params={sum(p.numel() for p in model.parameters()):,}")

    # ── DAgger loop ──
    all_data = []
    best_rate = 0.0

    for dagger_iter in range(args.dagger_iters):
        print(f"\n{'='*60}")
        print(f"DAgger iter {dagger_iter+1}/{args.dagger_iters}")

        # Collect data
        n_collect = args.collect_episodes // args.dagger_iters
        print(f"\n[Collect] {n_collect} episodes (BC policy, no residual yet)...")
        new_data = collect_data(env, pi05, pre, post, vis_dim, device, n_collect)
        all_data.extend(new_data)

        # Train
        print(f"\n[Train] {len(all_data)} total samples, {args.train_epochs} epochs...")
        model = train_residual(model, all_data, device,
                               epochs=args.train_epochs,
                               batch_size=args.batch_size,
                               lr=args.lr)

        # Eval
        print(f"\n[Eval] {args.eval_episodes} episodes...")
        rate = eval_residual(model, env, pi05, pre, post, vis_dim, device, args.eval_episodes)

        if rate > best_rate:
            best_rate = rate
            torch.save(model.state_dict(), args.save)
            print(f"  Saved best model (rate={rate:.1f}%)")

    env.close()
    print(f"\nBest eval rate: {best_rate:.1f}%")
    print(f"Model saved to {args.save}")


if __name__ == "__main__":
    main()
