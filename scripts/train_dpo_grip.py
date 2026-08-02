"""Grip-only DPO: train a grip policy via pairwise preference.

BC controls XYZ (frozen PI0.5). DPO trains a tiny grip policy.
Pairs: same EE+cube position → prefer success grip over failure grip.

Usage:
    MUJOCO_GL=egl python train_dpo_grip.py \
        --checkpoint <path> --collect_eps 500 --dpo_steps 5000
"""

import argparse, warnings, random
from collections import defaultdict
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


def grip_features(obs, env):
    """11D context for grip decision."""
    ee = obs["observation.state"][8:11]
    cube = env.cube_position
    grip = obs["observation.state"][7]
    return np.concatenate([
        ee, cube, [grip],
        [np.linalg.norm(ee[:2] - cube[:2])],  # xy dist
        ee - cube,                              # rel pos (3)
    ]).astype(np.float32)


def collect_data(pi05, pre, post, env, device, n_eps):
    """Collect (state_feat, bc_grip, success) per frame."""
    data, succ_cnt = [], 0
    succ_min_dist, fail_min_dist = [], []
    for ep in range(n_eps):
        env.reset()
        obs = env.get_obs_pi05()
        done = False
        ep_min_dist = float('inf')
        while not done:
            batch = pre(build_batch(obs, device))
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = pi05.select_action(batch)
                bc = post(raw).squeeze(0).cpu().numpy()
            feat = grip_features(obs, env)
            rl = np.array([bc[0], bc[1], bc[2], bc[6]])
            _, _, t, tr, _ = env.step(rl)
            d3d = np.linalg.norm(env.ee_position - env.cube_position)
            ep_min_dist = min(ep_min_dist, d3d)
            data.append({"feat": feat, "grip_bc": bc[6],
                         "success": env._success})
            done = t or tr
            obs = env.get_obs_pi05()
        if env._success:
            succ_cnt += 1
            succ_min_dist.append(ep_min_dist)
        else:
            fail_min_dist.append(ep_min_dist)
        if ep % 50 == 0 or ep < 5:
            print(f"  ep {ep:4d}: succ={env._success}, total={succ_cnt}/{ep+1}")
    print(f"\nCollected {n_eps} eps: {succ_cnt} success ({succ_cnt/n_eps*100:.0f}%)")
    if succ_min_dist:
        print(f"  Success min 3D dist: mean={np.mean(succ_min_dist):.4f} std={np.std(succ_min_dist):.4f} min={np.min(succ_min_dist):.4f}")
    if fail_min_dist:
        print(f"  Failure min 3D dist: mean={np.mean(fail_min_dist):.4f} std={np.std(fail_min_dist):.4f} min={np.min(fail_min_dist):.4f}")
    return data


class GripDPO(nn.Module):
    """11D → grip delta. Init=0 → pure BC. Residual approach."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, bc_grip):
        delta = 0.2 * torch.tanh(self.net(x))  # ±0.2 max correction
        return bc_grip + delta  # residual: BC + delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--collect_eps", type=int, default=500)
    parser.add_argument("--dpo_steps", type=int, default=5000)
    parser.add_argument("--save", default="./grip_dpo.pt")
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

    # Phase 1
    print(f"\n=== Phase 1: Collecting {args.collect_eps} episodes ===")
    data = collect_data(pi05, pre, post, env, device, args.collect_eps)

    # Phase 2: Build grip pairs
    print("\n=== Phase 2: Building grip preference pairs ===")
    # Filter: only near-grasp frames (EE-cube < 15cm)
    def near_grasp(d):
        ee, cube = d["feat"][:3], d["feat"][3:6]
        d3d = np.linalg.norm(ee - cube)
        return 0.03 < d3d < 0.10  # only near-grasp, not approach or lift

    goods = [d for d in data if d["success"] and near_grasp(d)]
    bads = [d for d in data if not d["success"] and near_grasp(d)]
    print(f"  near-grasp: {len(goods)}G + {len(bads)}B")

    # Nearest-neighbor pairing: for each good frame, find closest bad frame
    pairs = []
    for g in random.sample(goods, min(2000, len(goods))):
        # Closest bad frame by EE distance
        g_ee = g["feat"][:3]
        # Find bad frame with EE within 3cm, same cube position
        g_ee, g_cube = g["feat"][:3], g["feat"][3:6]
        candidates = [b for b in bads
                      if np.linalg.norm(b["feat"][:3] - g_ee) < 0.03
                      and np.linalg.norm(b["feat"][3:6] - g_cube) < 0.03]
        if not candidates: continue
        best_b = min(candidates, key=lambda b: np.linalg.norm(b["feat"][:3] - g_ee))
        pairs.append({"feat": g["feat"], "bc_grip": g["grip_bc"],
                      "good_grip": g["grip_bc"], "bad_grip": best_b["grip_bc"]})
    if pairs:
        diffs = [abs(p["good_grip"] - p["bad_grip"]) for p in pairs]
        print(f"Built {len(pairs)} grip pairs.")
        print(f"  |good-bad| mean={np.mean(diffs):.4f} median={np.median(diffs):.4f} max={np.max(diffs):.4f}")
        print(f"  good grip dist: mean={np.mean([p['good_grip'] for p in pairs]):.3f} std={np.std([p['good_grip'] for p in pairs]):.3f}")
        print(f"  bad grip dist:  mean={np.mean([p['bad_grip'] for p in pairs]):.3f} std={np.std([p['bad_grip'] for p in pairs]):.3f}")
    else:
        print("Built 0 pairs.")

    if len(pairs) < 10:
        print("Not enough pairs.")
        env.close()
        return

    # Normalize features
    all_feat = np.array([p["feat"] for p in pairs], dtype=np.float32)
    f_mean, f_std = all_feat.mean(axis=0), all_feat.std(axis=0) + 1e-8

    # Phase 3: DPO training
    print(f"\n=== Phase 3: DPO training ({args.dpo_steps} steps) ===")
    model = GripDPO().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(args.dpo_steps):
        batch = random.sample(pairs, min(256, len(pairs)))
        F = torch.tensor((np.array([p["feat"] for p in batch]) - f_mean) / f_std,
                         dtype=torch.float32, device=device)
        BC = torch.tensor([p["bc_grip"] for p in batch],
                          dtype=torch.float32, device=device).unsqueeze(1)
        G = torch.tensor([p["good_grip"] for p in batch],
                         dtype=torch.float32, device=device).unsqueeze(1)
        B = torch.tensor([p["bad_grip"] for p in batch],
                         dtype=torch.float32, device=device).unsqueeze(1)

        pred = model(F, BC)  # residual: BC + delta
        std = 0.2
        # Trained policy log-probs
        logp_good = -((pred - G) ** 2).sum(dim=-1) / (2 * std * std)
        logp_bad  = -((pred - B) ** 2).sum(dim=-1) / (2 * std * std)
        # Reference policy (BC, delta=0): pred_ref = BC
        ref_good = -((BC - G) ** 2).sum(dim=-1) / (2 * std * std)
        ref_bad  = -((BC - B) ** 2).sum(dim=-1) / (2 * std * std)

        # Standard DPO with reference
        log_ratio = args.beta * ((logp_good - ref_good) - (logp_bad - ref_bad))
        loss = -torch.nn.functional.logsigmoid(log_ratio).mean()
        loss += 0.01 * ((pred - BC) ** 2).mean()  # KL: stay near BC

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0 or step == args.dpo_steps - 1:
            print(f"  step {step:5d}: loss={loss.item():.4f}, |pred|={pred.abs().mean().item():.4f}")

    torch.save({"model_state": model.state_dict(),
                "f_mean": f_mean, "f_std": f_std}, args.save)
    print(f"Saved: {args.save}")
    env.close()


if __name__ == "__main__":
    main()
