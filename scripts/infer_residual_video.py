"""Server-side inference with residual policy + video recording.

Usage (on server):
    MUJOCO_GL=egl python infer_residual_video.py \
        --checkpoint /root/autodl-tmp/output_my_data/checkpoints/020000/pretrained_model \
        --residual-model ./residual_net.pt \
        --episodes 1 --max-steps 100 --output ./output.mp4
"""

import argparse, warnings, time
import numpy as np
import torch, torch.nn as nn

# ── ResidualNet (must match train_residual.py) ──
class ResidualNet(nn.Module):
    def __init__(self, vis_dim):
        super().__init__()
        self.vis_proj = nn.Linear(vis_dim, 64)
        self.net = nn.Sequential(
            nn.Linear(64 + 7 + 3, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, vis_feat, state7, bc_xyz):
        v = self.vis_proj(vis_feat)
        x = torch.cat([v, state7, bc_xyz], dim=-1)
        raw = torch.tanh(self.net(x))
        scale = torch.tensor([0.02, 0.02, 0.005], device=x.device)
        return raw * scale


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
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--output", default="./output.mp4")
    parser.add_argument("--fps", type=int, default=15)
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

    # ── Hook action_in_proj ──
    _feat = None
    def _hook(m, inp, out):
        nonlocal _feat
        _feat = out.detach()
    vis_dim = 512
    for n, m in pi05.named_modules():
        if n.endswith("action_in_proj"):
            m.register_forward_hook(_hook)
            if hasattr(m, 'out_features'):
                vis_dim = m.out_features
            print(f"  Hooked {n}, vis_dim={vis_dim}")
            break

    pre, post = make_pre_post_processors(policy_cfg=pi05.config, pretrained_path=args.checkpoint)
    print("PI0.5 loaded.")

    # ── Load residual model ──
    residual = ResidualNet(vis_dim).to(device)
    residual.load_state_dict(torch.load(args.residual_model, map_location=device))
    residual.eval()
    print(f"Residual model loaded.")

    # ── Env ──
    import mujoco
    from panda_rl_env import PandaRLEnv
    env = PandaRLEnv()

    # Setup renderer for video
    render_w, render_h = 640, 480
    renderer = mujoco.Renderer(env.model, render_h, render_w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = np.array([0.5, -0.1, 0.25])
    cam.distance = 0.60
    cam.azimuth = 160
    cam.elevation = -25.0

    frames = []

    for ep in range(args.episodes):
        env.reset()
        obs = env.get_obs_pi05()
        step = 0
        ep_frames = 0

        print(f"\nEpisode {ep+1}/{args.episodes}")
        cube = env.cube_position
        print(f"  Cube at: ({cube[0]:.2f}, {cube[1]:.2f}, {cube[2]:.2f})")

        while step < args.max_steps:
            # PI0.5 inference
            batch = pre(build_batch(obs, device))
            _feat = None
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = pi05.select_action(batch)
                bc = post(raw).squeeze(0).cpu().numpy()

            # Get vision features + compute residual
            vis_raw = _feat.squeeze(0) if _feat is not None else torch.zeros(vis_dim, device=device)
            if vis_raw.dim() == 2:
                vis_raw = vis_raw.mean(dim=0)
            vis_feat = vis_raw if vis_raw.dim() == 1 else vis_raw.squeeze(0)

            ee = env.ee_position
            cube = env.cube_position
            state7 = torch.tensor(np.concatenate([ee, cube, [env.gripper_width]]),
                                  dtype=torch.float32, device=device).unsqueeze(0)
            bc_xyz = torch.tensor(bc[:3], dtype=torch.float32, device=device).unsqueeze(0)
            vis_t = vis_feat.unsqueeze(0) if vis_feat.dim() == 1 else vis_feat

            with torch.no_grad():
                delta = residual(vis_t, state7, bc_xyz).squeeze(0).cpu().numpy()

            bc_c = bc.copy()
            bc_c[:3] = bc[:3] + delta
            rl = np.array([bc_c[0], bc_c[1], bc_c[2], bc[6]])

            if step < 10 or step % 15 == 0:
                dist = float(np.linalg.norm(ee - cube))
                print(f"  step {step:3d} | dist={dist*100:.1f}cm | "
                      f"BC({bc[0]:+.3f},{bc[1]:+.3f},{bc[2]:+.3f}) "
                      f"Δ({delta[0]:+.4f},{delta[1]:+.4f},{delta[2]:+.4f})")

            _, _, terminated, truncated, _ = env.step(rl)
            done = terminated or truncated
            obs = env.get_obs_pi05()

            # Render frame
            renderer.update_scene(env.data, camera=cam)
            frame = renderer.render().copy()  # (H, W, 3) uint8 RGB
            frames.append(frame)
            ep_frames += 1
            step += 1

            if done:
                final_dist = float(np.linalg.norm(env.ee_position - env.cube_position))
                status = "✓ SUCCESS" if env._success else f"✗ (dist={final_dist*100:.1f}cm)"
                print(f"  {status} | steps={step} | frames={ep_frames}")
                break

    env.close()
    renderer.close()

    # ── Save video ──
    print(f"\nSaving {len(frames)} frames to {args.output} ...")
    try:
        import cv2
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Video saved: {args.output}")
    except ImportError:
        try:
            import imageio
            imageio.mimsave(args.output, frames, fps=args.fps)
            print(f"Video saved: {args.output}")
        except Exception:
            np.save(args.output.replace(".mp4", ".npy"), np.stack(frames))
            print(f"Saved frames as .npy: {args.output.replace('.mp4', '.npy')}")
            print("  Install opencv-python or imageio[ffmpeg] for mp4 output")


if __name__ == "__main__":
    main()
