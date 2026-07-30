"""Test: compute predicted EE target positions from action chunk.

Key insight: action chunk (50 steps) cumulatively predicts where the
gripper should move. If the model learns the correct spatial mapping,
the final predicted EE y should be close to the cube y.
"""

import base64, io, json, sys, urllib.request
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from vla_inference.env.panda_joint_env import PandaJointEnv
import mujoco

SERVER = "http://localhost:8765/predict"

def query(front_b64, wrist_b64, state_list):
    payload = json.dumps({
        "front_image": front_b64, "wrist_image": wrist_b64,
        "state": state_list, "task": "move to the red cube and pick it up",
    }).encode()
    req = urllib.request.Request(SERVER, data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return np.array(json.loads(resp.read())["action"])

def to_b64(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def set_cube(env, x, y, z=0.03):
    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
    jnt_id = env.model.body_jntadr[body_id]
    adr = env.model.jnt_qposadr[jnt_id]
    env.data.qpos[adr:adr+3] = [x, y, z]
    mujoco.mj_forward(env.model, env.data)


def main():
    env = PandaJointEnv(render_mode="rgb_array", scene="task")
    obs = env.reset()
    home_state = obs["observation.state"].tolist()
    current_ee = np.array(home_state[8:11])

    print("=" * 70)
    print("Predicted EE target positions vs cube positions")
    print("=" * 70)

    positions = {
        "far_left ": (0.45, -0.40),
        "center   ": (0.45, 0.00),
        "far_right": (0.45, 0.40),
    }

    for label, (cx, cy) in positions.items():
        set_cube(env, cx, cy)
        front = to_b64(env._get_obs()["observation.images.front"])
        wrist = to_b64(env._get_obs()["observation.images.wrist"])

        # Get full action chunk (50 steps)
        chunk = []
        for _ in range(50):
            chunk.append(query(front, wrist, home_state))
        chunk = np.array(chunk)  # (50, 7)

        # Predicted EE trajectory: cumsum of deltas
        cum_delta = np.cumsum(chunk[:, :3], axis=0)  # (50, 3)
        pred_traj = current_ee + cum_delta  # (50, 3)

        # Print at key chunk positions
        print(f"\n  cube at {label.strip()} (y={cy:+.2f}):")
        print(f"  {'t':>3s}  {'pred_x':>8s}  {'pred_y':>8s}  {'pred_z':>8s}  {'cube_y':>8s}  {'dy_err':>8s}")
        for t in [0, 5, 10, 20, 30, 40, 49]:
            py = pred_traj[t, 1]
            err = py - cy
            print(f"  {t:3d}  {pred_traj[t,0]:8.3f}  {py:8.3f}  {pred_traj[t,2]:8.3f}  {cy:8.2f}  {err:8.3f}")

        # Final predicted target (end of chunk)
        final_y = pred_traj[-1, 1]
        final_err = final_y - cy
        print(f"  Final pred_y={final_y:.3f}, cube_y={cy:.2f}, error={final_err:.3f} ({final_err*100:.1f}cm)")

        # Average dy direction in chunk
        dy_mean = chunk[:, 1].mean()
        print(f"  Chunk dy mean: {dy_mean:+.4f}  (should be {'positive' if cy > 0 else 'negative' if cy < 0 else '~zero'})")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
