"""Inference script: run BC + residual policy with per-step diagnostics.

Usage (on server):
    MUJOCO_GL=egl python infer_residual.py \
        --checkpoint /root/autodl-tmp/output_my_data/checkpoints/020000/pretrained_model \
        --residual-model ./residual_net_v1.pt --episodes 3
"""

import argparse, warnings
import numpy as np
import torch
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--residual-model", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
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

    # ── Load residual model ──
    from train_residual import ResidualNet
    residual = ResidualNet(vis_dim).to(device)
    residual.load_state_dict(torch.load(args.residual_model, map_location=device))
    residual.eval()
    print(f"Residual model loaded from {args.residual_model}")

    env = PandaRLEnv()
    successes = 0

    for ep in range(args.episodes):
        env.reset()
        obs = env.get_obs_pi05()
        done = False
        step = 0

        print(f"\n{'='*70}")
        print(f"Episode {ep+1}/{args.episodes}")
        print(f"  Cube at: {env.cube_position}")
        print(f"{'Step':>5} {'Dist(cm)':>8} {'BC(dx,dy,dz)':>22} {'Δ(dx,dy,dz)':>22} {'Action(dx,dy,dz)':>24} {'Grip':>5}")
        print(f"{'─'*5} {'─'*8} {'─'*22} {'─'*22} {'─'*24} {'─'*5}")

        while not done and step < args.max_steps:
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
            dist = float(np.linalg.norm(ee - cube))
            state7 = torch.tensor(np.concatenate([ee, cube, [env.gripper_width]]),
                                  dtype=torch.float32, device=device).unsqueeze(0)
            bc_xyz = torch.tensor(bc[:3], dtype=torch.float32, device=device).unsqueeze(0)
            vis_t = vis_feat.unsqueeze(0) if vis_feat.dim() == 1 else vis_feat

            with torch.no_grad():
                delta = residual(vis_t, state7, bc_xyz).squeeze(0).cpu().numpy()

            bc_c = bc.copy()
            bc_c[:3] = bc[:3] + delta
            rl = np.array([bc_c[0], bc_c[1], bc_c[2], bc[6]])
            _, _, terminated, truncated, _ = env.step(rl)
            done = terminated or truncated
            obs = env.get_obs_pi05()

            if step < 20 or step % 5 == 0 or done:
                print(f"{step:5d} {dist*100:8.1f} "
                      f"({bc[0]:+6.3f},{bc[1]:+6.3f},{bc[2]:+6.3f})  "
                      f"({delta[0]:+6.3f},{delta[1]:+6.3f},{delta[2]:+6.3f})  "
                      f"({bc_c[0]:+6.3f},{bc_c[1]:+6.3f},{bc_c[2]:+6.3f})  "
                      f"{bc_c[6]:5.2f}")

            step += 1

        if env._success:
            successes += 1
            print(f"  ✓ SUCCESS at step {step}")
        elif env._grasped:
            print(f"  ✗ GRASPED but not lifted (step {step})")
        else:
            final_dist = float(np.linalg.norm(env.ee_position - env.cube_position))
            print(f"  ✗ FAILED | final dist={final_dist*100:.1f}cm | steps={step}")

    env.close()
    print(f"\n{'='*70}")
    print(f"Result: {successes}/{args.episodes} = {successes/args.episodes*100:.0f}%")


if __name__ == "__main__":
    main()
