"""Last-mile DPO: positional refinement in final 5cm via preference learning.

Collects BC trajectories. For frames within 5cm of cube:
  success frames → good position correction
  failure frames → bad position correction
DPO learns to prefer successful corrections.

Usage:
    MUJOCO_GL=egl python train_dpo_lastmile.py \
        --checkpoint <path> --collect_eps 500 --dpo_steps 5000
"""

import argparse, warnings, random
import numpy as np
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


def collect_data(pi05, pre, post, env, device, n_eps):
    """Collect (state_feat, bc_xyz, success) for last-mile frames (dist<5cm)."""
    data, succ_cnt = [], 0
    for ep in range(n_eps):
        env.reset()
        obs = env.get_obs_pi05()
        done = False
        ep_dist = float('inf')
        ep_frames = []  # collect frames, label at episode end
        while not done:
            batch = pre(build_batch(obs, device))
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = pi05.select_action(batch)
                bc = post(raw).squeeze(0).cpu().numpy()
            ee, cube = env.ee_position, env.cube_position
            dist = float(np.linalg.norm(ee - cube))
            ep_dist = min(ep_dist, dist)

            rl = np.array([bc[0], bc[1], bc[2], bc[6]])
            _, _, t, tr, _ = env.step(rl)
            done = t or tr

            if dist < 0.05:
                feat = np.concatenate([ee, cube, [env.gripper_width]]).astype(np.float32)
                # Store frame — success label assigned at episode end
                ep_frames.append({"feat": feat, "action_xyz": bc[:3].copy()})
            obs = env.get_obs_pi05()
        if env._success: succ_cnt += 1
        # Assign episode-level success to all last-mile frames
        for f in ep_frames:
            f["success"] = env._success
        data.extend(ep_frames)
        if ep % 50 == 0 or ep < 5:
            print(f"  ep {ep:4d}: min_dist={ep_dist:.3f} succ={env._success} total={succ_cnt}/{ep+1}")
    print(f"\nCollected {n_eps} eps: {succ_cnt} success ({succ_cnt/n_eps*100:.0f}%)")
    print(f"  Last-mile frames: {len(data)}")
    # Save data to disk so we don't lose it on crash
    np.savez_compressed("/tmp/lastmile_data.npz",
                        feat=np.array([d["feat"] for d in data]),
                        action_xyz=np.array([d["action_xyz"] for d in data]),
                        success=np.array([d["success"] for d in data]))
    print("  Saved to /tmp/lastmile_data.npz")
    return data


class LastMileDPO(nn.Module):
    """7D state → Δxyz action correction (meters). Init=0 → pure BC."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat):
        return 0.005 * torch.tanh(self.net(feat))  # Δxyz, ±5mm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--collect_eps", type=int, default=500)
    parser.add_argument("--dpo_steps", type=int, default=5000)
    parser.add_argument("--save", default="./lastmile_dpo.pt")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Device: {device}, DPO beta={args.beta}")

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

    # Phase 1: Collect
    print(f"\n=== Phase 1: Collecting {args.collect_eps} episodes ===")
    data = collect_data(pi05, pre, post, env, device, args.collect_eps)

    # Phase 2: Build pairs
    print("\n=== Phase 2: Building last-mile preference pairs ===")
    goods = [d for d in data if d["success"]]
    bads = [d for d in data if not d["success"]]
    print(f"  {len(goods)}G + {len(bads)}B last-mile frames")

    pairs = []
    # Pair: same state, prefer action_xyz from success over action_xyz from failure
    for g in random.sample(goods, min(2000, len(goods))):
        b = min(bads, key=lambda x: np.linalg.norm(x["feat"][:3] - g["feat"][:3]))
        pairs.append({"feat": g["feat"],
                       "good_xyz": g["action_xyz"],   # displacement → success
                       "bad_xyz": b["action_xyz"]})    # displacement → failure

    if len(pairs) < 10:
        print("Not enough pairs.")
        env.close()
        return
    print(f"  Built {len(pairs)} pairs")

    # Phase 3: DPO training
    print(f"\n=== Phase 3: DPO training ({args.dpo_steps} steps) ===")
    model = LastMileDPO().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(args.dpo_steps):
        batch = random.sample(pairs, min(256, len(pairs)))
        F = torch.tensor(np.array([p["feat"] for p in batch]),
                         dtype=torch.float32, device=device)
        G = torch.tensor(np.array([p["good_xyz"] for p in batch]),
                         dtype=torch.float32, device=device)
        B = torch.tensor(np.array([p["bad_xyz"] for p in batch]),
                         dtype=torch.float32, device=device)

        # pred, G, B are ALL xyz displacements (meters). Same space.
        pred = model(F)  # (B, 3) — Δxyz
        std = 0.005  # matches ±5mm delta range
        logp_good = -((pred - G) ** 2).sum(dim=-1) / (2 * std * std)
        logp_bad  = -((pred - B) ** 2).sum(dim=-1) / (2 * std * std)
        ref_good = -(G ** 2).sum(dim=-1) / (2 * std * std)
        ref_bad  = -(B ** 2).sum(dim=-1) / (2 * std * std)

        log_ratio = args.beta * ((logp_good - ref_good) - (logp_bad - ref_bad))
        loss = -torch.nn.functional.logsigmoid(log_ratio).mean()
        loss += 0.01 * (pred ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0 or step == args.dpo_steps - 1:
            print(f"  step {step:5d}: loss={loss.item():.4f}, |pred|={pred.abs().mean().item():.5f}")

    torch.save({"model_state": model.state_dict()}, args.save)
    print(f"Saved: {args.save}")
    env.close()


if __name__ == "__main__":
    main()
