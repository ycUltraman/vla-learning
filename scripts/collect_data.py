"""Collect pick-and-place demonstration data in LeRobot v3.0 format.

Task: pick up the RED block and place it onto the pad.
Each episode randomizes which of 3 positions the red block starts at.

Uses PandaJointEnv (joint-space control) with a DLS IK controller
to move the end-effector through pick-and-place waypoints.

Output: ./collected_data_vla/  (LeRobot v3.0 format, ready for training)
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from vla_inference.env.panda_joint_env import PandaJointEnv

# ── Scene constants (must match scene_task.xml) ────────────────
CUBE_HALF = 0.03  # half-size of cube geom
PAD_POS = np.array([0.45, -0.15, 0.002])
PAD_HALF_Z = 0.025
PAD_CENTER_Z = PAD_POS[2] + PAD_HALF_Z  # top surface of pad

# Three possible cube start positions
POSITIONS = [
    np.array([0.35, 0.05, CUBE_HALF]),  # pos 0
    np.array([0.45, 0.05, CUBE_HALF]),  # pos 1
    np.array([0.55, 0.08, CUBE_HALF]),  # pos 2
]

# ── IK solver ──────────────────────────────────────────────────

def compute_ee_jacobian(env: PandaJointEnv) -> np.ndarray:
    """3x7 translational Jacobian at EE site."""
    jacp = np.zeros((3, env.model.nv))
    jacr = np.zeros((3, env.model.nv))
    hand_id = env._hand_body_id
    mujoco.mj_jacBody(env.model, env.data, jacp, jacr, hand_id)
    # EE is hand + offset; approximate with body Jacobian
    return jacp[:, :7].copy()


def ee_delta_to_joint_delta(env: PandaJointEnv, ee_delta: np.ndarray) -> np.ndarray:
    """DLS IK: convert desired EE delta → joint delta (7D)."""
    J = compute_ee_jacobian(env)  # 3x7
    damping = 0.05
    # dq = J^T (J J^T + λ²I)^(-1) dx
    A = J @ J.T + damping**2 * np.eye(3)
    dq = J.T @ np.linalg.solve(A, ee_delta)
    return dq


# ── Episode runner ─────────────────────────────────────────────

def run_episode(env: PandaJointEnv, red_pos_idx: int) -> list[dict]:
    """
    Execute one pick-and-place episode.
    Returns list of per-step dicts: {obs, action}.
    """
    _ = env.reset()
    frames = []
    red_pos = POSITIONS[red_pos_idx]
    speed = 0.04  # EE step size (meters)

    # Phase 1: Move to above red block
    for _ in range(30):
        target = np.array([red_pos[0], red_pos[1], 0.12])
        ee_err = target - env.ee_position
        if np.linalg.norm(ee_err) < 0.01:
            break
        dq = ee_delta_to_joint_delta(env, ee_err * speed)
        action = _build_action(env, dq, gripper_close=False)
        obs = env.step(action)
        frames.append({"obs": _copy_obs(obs), "action": action.copy()})

    # Phase 2: Descend to block, close gripper
    for _ in range(20):
        target = np.array([red_pos[0], red_pos[1], 0.065])
        ee_err = target - env.ee_position
        if np.linalg.norm(ee_err) < 0.005:
            break
        dq = ee_delta_to_joint_delta(env, ee_err * speed * 0.5)
        action = _build_action(env, dq, gripper_close=False)
        obs = env.step(action)
        frames.append({"obs": _copy_obs(obs), "action": action.copy()})

    # Phase 3: Close gripper (hold for a few steps)
    for _ in range(10):
        dq = np.zeros(7)
        action = _build_action(env, dq, gripper_close=True)
        obs = env.step(action)
        frames.append({"obs": _copy_obs(obs), "action": action.copy()})

    # Phase 4: Lift
    for _ in range(20):
        target = np.array([red_pos[0], red_pos[1], 0.12])
        ee_err = target - env.ee_position
        if np.linalg.norm(ee_err) < 0.02:
            break
        dq = ee_delta_to_joint_delta(env, ee_err * speed)
        action = _build_action(env, dq, gripper_close=True)
        obs = env.step(action)
        frames.append({"obs": _copy_obs(obs), "action": action.copy()})

    # Phase 5: Move to above pad
    pad_above = np.array([PAD_POS[0], PAD_POS[1], 0.12])
    for _ in range(40):
        ee_err = pad_above - env.ee_position
        if np.linalg.norm(ee_err) < 0.01:
            break
        dq = ee_delta_to_joint_delta(env, ee_err * speed)
        action = _build_action(env, dq, gripper_close=True)
        obs = env.step(action)
        frames.append({"obs": _copy_obs(obs), "action": action.copy()})

    # Phase 6: Descend to pad
    for _ in range(20):
        target = np.array([PAD_POS[0], PAD_POS[1], 0.045])
        ee_err = target - env.ee_position
        if np.linalg.norm(ee_err) < 0.005:
            break
        dq = ee_delta_to_joint_delta(env, ee_err * speed * 0.5)
        action = _build_action(env, dq, gripper_close=True)
        obs = env.step(action)
        frames.append({"obs": _copy_obs(obs), "action": action.copy()})

    # Phase 7: Open gripper
    for _ in range(10):
        dq = np.zeros(7)
        action = _build_action(env, dq, gripper_close=False)
        obs = env.step(action)
        frames.append({"obs": _copy_obs(obs), "action": action.copy()})

    # Phase 8: Retract
    for _ in range(15):
        target = np.array([PAD_POS[0], PAD_POS[1], 0.15])
        ee_err = target - env.ee_position
        if np.linalg.norm(ee_err) < 0.03:
            break
        dq = ee_delta_to_joint_delta(env, ee_err * speed)
        action = _build_action(env, dq, gripper_close=False)
        obs = env.step(action)
        frames.append({"obs": _copy_obs(obs), "action": action.copy()})

    return frames


# ── Helpers ────────────────────────────────────────────────────

def _build_action(env: PandaJointEnv, dq: np.ndarray, gripper_close: bool) -> np.ndarray:
    """Build 8D joint action from joint delta + gripper command."""
    current_joints = env.joint_positions
    target_joints = np.clip(current_joints + dq, -2.8, 2.8)
    gripper_cmd = 1.0 if gripper_close else 0.0
    return np.concatenate([target_joints, [gripper_cmd]]).astype(np.float64)


def _copy_obs(obs: dict) -> dict:
    """Deep-copy observation dict for storage."""
    return {
        "observation.state": obs["observation.state"].copy(),
        "observation.images.front": obs["observation.images.front"].copy(),
        "observation.images.wrist": obs["observation.images.wrist"].copy(),
    }


# ── Main ───────────────────────────────────────────────────────

import mujoco

OUTPUT_DIR = Path(__file__).parent.parent / "collected_data_vla"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--render", action="store_true", help="Show MuJoCo viewer")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "videos").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    (output_dir / "meta").mkdir(exist_ok=True)

    render_mode = "human" if args.render else "rgb_array"
    env = PandaJointEnv(render_mode=render_mode, scene="task")

    all_episodes = []
    total_frames = 0
    episode_lengths = []
    task_idx = 0  # single task

    for ep in range(args.episodes):
        # Randomize: which position gets the red block
        red_pos_idx = np.random.randint(0, 3)

        print(f"Episode {ep}/{args.episodes}: red at pos {red_pos_idx} ... ", end="", flush=True)
        frames = run_episode(env, red_pos_idx)
        print(f"{len(frames)} steps")

        episode_lengths.append(len(frames))
        total_frames += len(frames)

        # Build episode dict
        episode_frames = []
        for i, f in enumerate(frames):
            episode_frames.append({
                "observation.state": f["obs"]["observation.state"].tolist(),
                "action": f["action"].tolist(),
                "observation.images.front": f["obs"]["observation.images.front"].tolist(),
                "observation.images.wrist": f["obs"]["observation.images.wrist"].tolist(),
                "timestamp": i / 30.0,  # 30 fps
                "episode_index": ep,
                "frame_index": i,
                "task_index": task_idx,
                "task": "Pick the red block and place it on the pad",
                "index": total_frames - len(frames) + i,
            })
        all_episodes.append(episode_frames)

    env.close()

    # Save in LeRobot v3.0 format
    print(f"\nSaving {total_frames} frames from {args.episodes} episodes ...")

    # meta/info.json
    info = {
        "codebase_version": "v3.0",
        "robot_type": "franka_panda",
        "total_episodes": args.episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "fps": 30,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [15]},
            "action": {"dtype": "float32", "shape": [8]},
            "observation.images.front": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.wrist": {"dtype": "video", "shape": [480, 640, 3]},
        },
        "splits": ["train"],
    }
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # meta/tasks.parquet → jsonl (simpler)
    with open(output_dir / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "Pick the red block and place it on the pad"}) + "\n")

    # meta/episodes.jsonl
    with open(output_dir / "meta" / "episodes.jsonl", "w") as f:
        for ep_idx, length in enumerate(episode_lengths):
            f.write(json.dumps({"episode_index": ep_idx, "length": length}) + "\n")

    # meta/stats.json
    # Collect all states and actions for mean/std
    all_states = []
    all_actions = []
    for ep_frames in all_episodes:
        for f in ep_frames:
            all_states.append(f["observation.state"])
            all_actions.append(f["action"])
    all_states = np.array(all_states)
    all_actions = np.array(all_actions)
    stats = {
        "observation.state": {
            "mean": all_states.mean(axis=0).tolist(),
            "std": all_states.std(axis=0).tolist(),
            "min": all_states.min(axis=0).tolist(),
            "max": all_states.max(axis=0).tolist(),
        },
        "action": {
            "mean": all_actions.mean(axis=0).tolist(),
            "std": all_actions.std(axis=0).tolist(),
            "min": all_actions.min(axis=0).tolist(),
            "max": all_actions.max(axis=0).tolist(),
        },
    }
    with open(output_dir / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved to {output_dir}")
    print(f"Episodes: {args.episodes}, Total frames: {total_frames}")
    print(f"Avg episode length: {total_frames / args.episodes:.0f} steps")


if __name__ == "__main__":
    main()
