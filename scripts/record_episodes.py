"""Record BC episodes with video + per-frame stats for manual labeling.

Usage (on server):
    MUJOCO_GL=egl python record_episodes.py \
        --checkpoint /root/autodl-tmp/output_my_data/checkpoints/020000/pretrained_model \
        --episodes 20 --output ./labeled_data
"""

import argparse, json, time, warnings
from pathlib import Path
import numpy as np
import torch
import cv2
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
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", default="./labeled_data")
    args = parser.parse_args()

    device = torch.device("cuda")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for ep in range(args.episodes):
        env.reset()
        obs = env.get_obs_pi05()
        done = False
        frames_data = []
        front_frames = []

        while not done:
            # Record frame
            ee = env.ee_position.copy()
            cube = env.cube_position.copy()
            grip = env.gripper_width

            frames_data.append({
                "ee": ee.tolist(),
                "cube": cube.tolist(),
                "grip": grip,
                "grasped_check": bool(
                    np.linalg.norm(ee[:2] - cube[:2]) < 0.05 and
                    abs(ee[2] - cube[2] - 0.03) < 0.05 and
                    grip < 0.04
                ),
            })
            front_frames.append(obs["observation.images.front"].copy())

            batch = preprocessor(build_batch(obs, device))
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = pi05.select_action(batch)
                action = postprocessor(raw).squeeze(0).cpu().numpy()

            rl_action = np.array([action[0], action[1], action[2], action[6]])
            _, _, terminated, truncated, _ = env.step(rl_action)
            done = terminated or truncated
            obs = env.get_obs_pi05()

        n = len(frames_data)
        # Stats for analysis
        cube_zs = [f["cube"][2] for f in frames_data]
        grip_vals = [f["grip"] for f in frames_data]
        ee_cube_xy = [np.linalg.norm(np.array(f["ee"][:2]) - np.array(f["cube"][:2]))
                       for f in frames_data]
        grasp_frames = sum(1 for f in frames_data if f["grasped_check"])

        stats = {
            "episode": ep,
            "n_frames": n,
            "max_cube_z": max(cube_zs),
            "final_cube_z": cube_zs[-1],
            "min_grip": min(grip_vals),
            "final_grip": grip_vals[-1],
            "min_ee_cube_xy": min(ee_cube_xy),
            "final_ee_cube_xy": ee_cube_xy[-1],
            "grasp_frames": grasp_frames,
            "grasp_ratio": grasp_frames / max(n, 1),
        }

        # Save video
        h, w = front_frames[0].shape[:2]
        video_path = str(out_dir / f"ep_{ep:03d}.mp4")
        vid = cv2.VideoWriter(video_path, fourcc, 10, (w, h))
        for f in front_frames:
            vid.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vid.release()

        # Save stats
        with open(out_dir / f"ep_{ep:03d}.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"Ep {ep:3d} | {n:3d}frames | max_cube_z={stats['max_cube_z']:.3f} | "
              f"min_grip={stats['min_grip']:.3f} | min_xy={stats['min_ee_cube_xy']:.3f} | "
              f"grasp_frames={grasp_frames}")

    env.close()

    # Summary
    all_stats = []
    for ep in range(args.episodes):
        with open(out_dir / f"ep_{ep:03d}.json") as f:
            all_stats.append(json.load(f))

    print(f"\nSummary stats for manual labeling:")
    print(f"{'ep':>4s} {'frames':>6s} {'max_cube_z':>10s} {'min_grip':>9s} {'min_xy':>7s} {'grasp_f':>7s}")
    for s in all_stats:
        print(f"{s['episode']:4d} {s['n_frames']:6d} {s['max_cube_z']:10.3f} {s['min_grip']:9.3f} {s['min_ee_cube_xy']:7.3f} {s['grasp_frames']:7d}")

    print(f"\nSaved to {out_dir}/")
    print("Watch videos and label each episode as success/failure.")
    print("Then share the labels and I'll calibrate the thresholds.")


if __name__ == "__main__":
    main()
