"""Standard DPO for PI0.5 residual fine-tuning.

Phase 1: Collect BC episodes, record (obs, bc_action, success).
Phase 2: Build preference pairs: same cube+similar EE → good_act > bad_act.
Phase 3: DPO train a small residual MLP on top of frozen PI0.5.

Usage:
    MUJOCO_GL=egl python train_dpo.py \
        --checkpoint <path> --collect_eps 500 --dpo_steps 5000
"""

import argparse, warnings, copy, random
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
    """Collect (state, bc_action, success, cube_init) per frame."""
    data, succ_count, fail_count = [], 0, 0
    for ep in range(n_eps):
        env.reset()
        obs = env.get_obs_pi05()
        done = False
        while not done:
            batch = pre(build_batch(obs, device))
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = pi05.select_action(batch)
                act = post(raw).squeeze(0).cpu().numpy()
            rl = np.array([act[0], act[1], act[2], act[6]])
            state_14 = np.concatenate([
                obs["observation.state"][:7],   # joints
                obs["observation.state"][8:11], # ee
                env.cube_position.copy(),        # cube
                [obs["observation.state"][7]],   # grip
            ]).astype(np.float32)
            _, _, t, tr, _ = env.step(rl)     # step first, then check success
            data.append({"state": state_14, "act": act.astype(np.float32),
                         "success": env._success,
                         "cube_init": env._init_cube[:2].round(2).tolist()})
            done = t or tr
            obs = env.get_obs_pi05()
        if env._success: succ_count += 1
        else: fail_count += 1
        if ep % 10 == 0:
            print(f"  ep {ep:4d}: steps={env._step_count}, succ={env._success}, "
                  f"total={succ_count}/{ep+1}")
    print(f"\nCollected {n_eps} eps: {succ_count} success + {fail_count} fail "
          f"= {succ_count/n_eps*100:.0f}%")
    return data


def build_pairs(data):
    """Build DPO preference pairs: (state, good_act, bad_act).

    For each cube position, pair frames with similar EE positions:
    one from a success episode (good), one from failure (bad).
    """
    # Group frames by cube_init
    from collections import defaultdict
    by_cube = defaultdict(lambda: {"good": [], "bad": []})
    for d in data:
        key = tuple(d["cube_init"])
        if d["success"]:
            by_cube[key]["good"].append(d)
        else:
            by_cube[key]["bad"].append(d)

    print(f"  by_cube keys: {list(by_cube.keys())}")
    for k, v in by_cube.items():
        print(f"    {k}: {len(v['good'])}G {len(v['bad'])}B")
    pairs = []
    for key, group in by_cube.items():
        goods, bads = group["good"], group["bad"]
        if not goods or not bads:
            continue
        # Filter: only frames near grasp zone (0.04 < ee-cube < 0.15)
        def in_zone(s):
            # XY distance only — Z can be 20-40cm during approach
            d = np.linalg.norm(s["state"][7:9] - s["state"][10:12])
            return 0.01 < d < 0.25
        n_g, n_b = len(goods), len(bads)
        goods = [x for x in goods if in_zone(x)]
        bads = [x for x in bads if in_zone(x)]
        print(f"  cube {key}: {n_g}G→{len(goods)}G {n_b}B→{len(bads)}B (zone filtered)")
        if not goods or not bads:
            continue

        for g in random.sample(goods, min(50, len(goods))):
            def dist(a, b):
                sa, sb = a["state"], b["state"]
                return (0.5*np.linalg.norm(sa[7:10]-sb[7:10]) +
                        0.3*np.linalg.norm(sa[10:13]-sb[10:13]) +
                        0.2*np.linalg.norm(sa[:7]-sb[:7]))
            best_bad = min(bads, key=lambda b: dist(g, b))
            pairs.append({
                "state": g["state"].copy(),
                "bc_act": g["act"].copy(),        # PI0.5 output at this state
                "good_act": g["act"].copy(),
                "bad_act": best_bad["act"].copy(),
            })
    print(f"Built {len(pairs)} preference pairs from {len(by_cube)} cube positions")
    return pairs


class ResidualDPO(nn.Module):
    """Small residual MLP: state(14D) + bc_action(7D) → delta(7D). Init=0."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(21, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 7),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state, bc_action):
        x = torch.cat([state, bc_action], dim=-1)
        return 0.1 * torch.tanh(self.net(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--collect_eps", type=int, default=500)
    parser.add_argument("--dpo_steps", type=int, default=5000)
    parser.add_argument("--save", default="./dpo_residual.pt")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Device: {device}, DPO beta={args.beta}")

    # Load PI0.5 (frozen BC)
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
    raw_data = collect_data(pi05, pre, post, env, device, args.collect_eps)

    # Phase 2: Build pairs
    print("\n=== Phase 2: Building preference pairs ===")
    pairs = build_pairs(raw_data)
    if len(pairs) < 10:
        print("Not enough pairs. Need more collection.")
        env.close()
        return

    # Normalize states
    all_states = np.array([p["state"] for p in pairs], dtype=np.float32)
    s_mean, s_std = all_states.mean(axis=0), all_states.std(axis=0) + 1e-8

    # Phase 3: DPO training
    print(f"\n=== Phase 3: DPO training ({args.dpo_steps} steps) ===")
    model = ResidualDPO().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    std = 0.1  # larger std = stable DPO gradient

    for step in range(args.dpo_steps):
        batch = random.sample(pairs, min(256, len(pairs)))
        S = torch.tensor((np.array([p["state"] for p in batch]) - s_mean) / s_std,
                         dtype=torch.float32, device=device)
        BC = torch.tensor(np.array([p["bc_act"] for p in batch]),
                         dtype=torch.float32, device=device)
        G = torch.tensor(np.array([p["good_act"] for p in batch]),
                         dtype=torch.float32, device=device)
        B = torch.tensor(np.array([p["bad_act"] for p in batch]),
                         dtype=torch.float32, device=device)

        # Residual on BC: state + bc_action → delta
        residual = model(S, BC)
        pred = BC + residual

        # Grip ×20: timing is the real issue (PPO proved XYZ is OK)
        weight = torch.tensor([0.5,0.5,0.5,1.0,1.0,1.0,20.0], device=device)
        # Trained policy log-probs
        logp_good = -(((pred - G) ** 2) * weight).sum(dim=-1) / (2 * std * std)
        logp_bad  = -(((pred - B) ** 2) * weight).sum(dim=-1) / (2 * std * std)
        # Reference policy (frozen BC, residual=0): pred_ref = BC
        ref_good = -(((BC - G) ** 2) * weight).sum(dim=-1) / (2 * std * std)
        ref_bad  = -(((BC - B) ** 2) * weight).sum(dim=-1) / (2 * std * std)

        # Standard DPO: prefer trained over reference for good, reverse for bad
        log_ratio = args.beta * ((logp_good - ref_good) - (logp_bad - ref_bad))
        dpo_loss = -torch.nn.functional.logsigmoid(log_ratio).mean()
        kl_reg = 0.01 * (residual ** 2).mean()  # tanh already bounds residual
        loss = dpo_loss + kl_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0 or step == args.dpo_steps - 1:
            residual_norm = residual.abs().mean().item()
            print(f"  step {step:5d}: dpo_loss={loss.item():.4f}, "
                  f"|residual|={residual_norm:.4f}")

    torch.save({"model_state": model.state_dict(),
                "s_mean": s_mean, "s_std": s_std}, args.save)
    print(f"Saved: {args.save}")
    env.close()


if __name__ == "__main__":
    main()
