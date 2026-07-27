"""Replay a collected episode in MuJoCo — verify recording fidelity.

Usage:
    python vla_inference/replay.py --ep 0      # replay episode 0
    python vla_inference/replay.py --ep 5 --fps 5  # slow motion
"""

import argparse, sys, time, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from vla_inference.env.panda_joint_env import PandaJointEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, default=0, help="Episode index to replay")
    parser.add_argument("--data", default="./collected_episodes", help="Data directory")
    parser.add_argument("--fps", type=int, default=10, help="Replay speed (fps)")
    args = parser.parse_args()

    ep_dir = Path(args.data) / f"ep_{args.ep:04d}"
    if not ep_dir.exists():
        print(f"Episode {args.ep} not found at {ep_dir}")
        sys.exit(1)

    # Load recorded data
    traj = np.load(ep_dir / "trajectory.npz", allow_pickle=True)
    states = traj["states"].astype(np.float32)
    actions = traj["actions"].astype(np.float32)

    # Load frame images for comparison
    front_npz = np.load(ep_dir / "images_front.npz", allow_pickle=True)
    wrist_npz = np.load(ep_dir / "images_wrist.npz", allow_pickle=True)
    recorded_front = front_npz[front_npz.files[0]].astype(np.uint8)
    recorded_wrist = wrist_npz[wrist_npz.files[0]].astype(np.uint8)

    with open(ep_dir / "meta.json") as f:
        meta = json.load(f)

    n_frames = len(actions)
    print(f"Episode {args.ep}: {n_frames} frames, task: {meta.get('task', '?')}")
    print(f"State shape: {states.shape}, Action shape: {actions.shape}")
    print(f"Replaying at {args.fps} fps...\n")
    print("action delta range:")
    print(f"  min: {actions[:, :3].min(axis=0)}")
    print(f"  max: {actions[:, :3].max(axis=0)}")
    print(f"  mean: {actions[:, :3].mean(axis=0)}\n")

    # Create env and reset to match the first state
    env = PandaJointEnv(render_mode="human", scene="task")
    obs = env.reset()

    print(f"Recorded initial state:")
    print(f"  state[0]:  joints={states[0][:7].round(3)}  grip={states[0][7]:.3f}  ee={states[0][8:11].round(3)}")
    print(f"  Current:   joints={obs['observation.state'][:7].round(3)}  grip={obs['observation.state'][7]:.3f}  ee={obs['observation.state'][8:11].round(3)}")

    step_interval = 1.0 / args.fps

    import cv2
    out_dir = Path("replay_frames")
    out_dir.mkdir(exist_ok=True)

    # Video writers: recorded vs replayed side-by-side
    h, w = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    front_vid = cv2.VideoWriter(str(out_dir / f"ep{args.ep}_front.mp4"), fourcc, args.fps, (w*2, h))
    wrist_vid = cv2.VideoWriter(str(out_dir / f"ep{args.ep}_wrist.mp4"), fourcc, args.fps, (w*2, h))

    replay_ee = np.zeros((n_frames, 3))
    try:
        for i in range(n_frames):
            t0 = time.perf_counter()
            t0 = time.perf_counter()
            obs = env.apply_ee_delta(actions[i])
            replay_ee[i] = obs["observation.state"][8:11]

            ee_err = np.linalg.norm(replay_ee[i] - states[i][8:11])

            if i % args.fps == 0 or i < 5:
                print(f"  [{i:3d}] grip={actions[i][6]:.1f} | "
                      f"ee={obs['observation.state'][8:11].round(3)} "
                      f"target={states[i][8:11].round(3)} "
                      f"ee_err={ee_err:.4f}")

            # Side-by-side frame: recorded | replayed
            for cam, rec, cur, vid in [
                ("front", recorded_front, obs["observation.images.front"], front_vid),
                ("wrist", recorded_wrist, obs["observation.images.wrist"], wrist_vid)]:
                cmp = np.concatenate([rec[i], cur], axis=1)
                vid.write(cv2.cvtColor(cmp, cv2.COLOR_RGB2BGR))

            elapsed = time.perf_counter() - t0
            if elapsed < step_interval:
                time.sleep(step_interval - elapsed)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()
        front_vid.release()
        wrist_vid.release()

    print(f"Done. Videos saved to {out_dir}/")

    # Compare EE trajectories
    print(f"\n{'='*60}")
    print("EE trajectory comparison (real vs replay)")
    print(f"{'='*60}")
    print(f"{'frame':>5} {'real_x':>8} {'replay_x':>8} {'real_y':>8} {'replay_y':>8} {'real_z':>8} {'replay_z':>8}")
    for i in [0, n_frames//4, n_frames//2, 3*n_frames//4, n_frames-1]:
        print(f"{i:5d} {states[i][8]:8.3f} {replay_ee[i][0]:8.3f} "
              f"{states[i][9]:8.3f} {replay_ee[i][1]:8.3f} "
              f"{states[i][10]:8.3f} {replay_ee[i][2]:8.3f}")
    # Print EE error stats
    ee_errs = np.linalg.norm(replay_ee - states[:, 8:11], axis=1)
    print(f"\nEE error: mean={ee_errs.mean():.4f} max={ee_errs.max():.4f}")


if __name__ == "__main__":
    main()
