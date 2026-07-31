"""Watch BC episodes in MuJoCo, record data, label success/failure manually.

Usage (LOCAL — need display):
    python scripts/label_episodes.py --episodes 10 --output labels.json

Each episode: watch the robot, then type 'y' (success) or 'n' (failure).
Saves per-frame data for analysis.
"""

import argparse, json, time
from pathlib import Path
import numpy as np

# Must be local — use the full import path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from vla_inference.env.panda_joint_env import PandaJointEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", default="./labels.json")
    args = parser.parse_args()

    env = PandaJointEnv(render_mode="human", scene="task")
    labels = []

    print(f"\n{'='*50}")
    print("Watch each episode in the MuJoCo viewer.")
    print("After the episode ends (or you close the viewer), label it:")
    print("  y = success (picked up cube)")
    print("  n = failure (did not complete)")
    print("  q = quit")
    print(f"{'='*50}\n")

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        frames = []

        print(f"\n--- Episode {ep} ---")
        print("Running... watch the viewer.")

        while not done:
            # Record frame data before stepping
            ee = env.ee_position.copy()
            cube_pos = env.data.xpos[
                env.model.body(
                    env.model.body_jntadr[
                        env.model.body("red_cube").id
                    ]
                ).id
            ].copy() if hasattr(env.model, 'body') else np.zeros(3)

            # Get cube position properly
            cube_id = env.model.body("red_cube").id
            cube_pos = env.data.xpos[cube_id].copy()
            grip = env.gripper_width

            frames.append({
                "ee": ee.tolist(),
                "cube": cube_pos.tolist(),
                "grip": float(grip),
            })

            # Use default BC action: PI0.5 isn't available locally,
            # so we use compute_target_joints + step with a simple
            # pre-recorded or random policy. Instead, let's use
            # the env's built-in behavior or load checkpoint.

            # For local testing without PI0.5, use a simple heuristic:
            # Move toward cube slowly
            target = cube_pos + np.array([0, 0, 0.06])  # grasp point above cube
            err = target - ee
            dist = np.linalg.norm(err)
            if dist < 0.005 and grip > 0.02:
                # Try to lift
                ee_delta = np.array([0, 0, 0.02])
                grip_cmd = 1.0
            elif dist < 0.02:
                ee_delta = err * 0.3
                grip_cmd = 1.0
            else:
                ee_delta = err * 0.1 / max(dist, 0.01)
                grip_cmd = 0.0

            env.compute_target_joints(ee_delta, damped=False)
            obs = env.step(np.concatenate([env._ee_target_joints, [grip_cmd]]))

            # Check if episode should end (timeout or success)
            cube_z = env.data.xpos[cube_id][2]
            if len(frames) > 500 or (grip < 0.02 and cube_z > 0.10):
                done = True

        print(f"Episode {ep} done. {len(frames)} frames.")

        while True:
            label = input("Success? (y/n/q): ").strip().lower()
            if label in ('y', 'n', 'q'):
                break
            print("  Please type y, n, or q")

        if label == 'q':
            break

        labels.append({
            "episode": ep,
            "label": "success" if label == 'y' else "failure",
            "n_frames": len(frames),
            "final_ee": frames[-1]["ee"],
            "final_cube": frames[-1]["cube"],
            "final_grip": frames[-1]["grip"],
            "frames": frames,  # full trajectory for analysis
        })
        print(f"  -> labeled as: {'SUCCESS' if label == 'y' else 'FAILURE'}")

    env.close()

    with open(args.output, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"\nSaved {len(labels)} labeled episodes to {args.output}")


if __name__ == "__main__":
    main()
