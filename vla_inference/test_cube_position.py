"""Test: same robot state, different cube positions → does action change?"""

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
    """Move red cube to position without changing robot state."""
    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
    jnt_id = env.model.body_jntadr[body_id]
    adr = env.model.jnt_qposadr[jnt_id]
    env.data.qpos[adr:adr+3] = [x, y, z]
    mujoco.mj_forward(env.model, env.data)


def main():
    env = PandaJointEnv(render_mode="rgb_array", scene="task")
    obs = env.reset()
    home_state = obs["observation.state"].tolist()

    print("=" * 60)
    print("Same robot state (home), different cube positions")
    print("=" * 60)

    results = {}
    positions = {
        "far_left ": (0.45, -0.40),
        "far_right": (0.45, 0.40),
    }
    first_action = None

    for label, (cx, cy) in positions.items():
        set_cube(env, cx, cy)
        front = to_b64(env._get_obs()["observation.images.front"])
        wrist = to_b64(env._get_obs()["observation.images.wrist"])
        # Query 50 steps to see full action chunk (PI0.5 uses action chunking)
        chunk = []
        for _ in range(50):
            a = query(front, wrist, home_state)
            chunk.append(a)
        chunk = np.array(chunk)
        action = chunk[0]  # first step
        results[label] = action

        if first_action is None:
            first_action = action.copy()

        diff = np.linalg.norm(chunk[:, :3] - chunk[0, :3], axis=1).mean()
        print(f"  cube at {label} ({cx:.2f},{cy:.2f}):")
        for t in [0, 10, 20, 30, 40]:
            a = chunk[t]
            print(f"    t={t:2d}: EE_delta=({a[0]:+.4f}, {a[1]:+.4f}, {a[2]:+.4f}) grip={a[6]:.1f}")
        print(f"    chunk dy mean: {chunk[:, 1].mean():+.4f}")

    print()
    a1 = results["far_left "][:3]
    a2 = results["far_right"][:3]
    total_diff = np.linalg.norm(a1 - a2)
    print(f"  Far left vs Far right EE_delta diff: {total_diff:.4f} (meters)")

    if total_diff > 0.01:
        print("CONCLUSION: Vision encoder learned spatial relationship ✓")
    else:
        print("CONCLUSION: Vision encoder did NOT learn spatial mapping ✗")

    # ── Test 2: State shortcut ──
    print()
    print("=" * 60)
    print("Test 2: Same image, different states → does action change?")
    print("=" * 60)
    set_cube(env, 0.45, -0.40)  # far left position
    front = to_b64(env._get_obs()["observation.images.front"])
    wrist = to_b64(env._get_obs()["observation.images.wrist"])
    a_real = query(front, wrist, home_state)
    print(f"  Real state → EE_delta(dx,dy,dz)=({a_real[0]:+.4f}, {a_real[1]:+.4f}, {a_real[2]:+.4f})")

    zero_state = [0.0]*15
    a_zero = query(front, wrist, zero_state)
    d_zero = np.linalg.norm(a_zero[:3] - a_real[:3])
    print(f"  Zero state → EE_delta=({a_zero[0]:+.4f}, {a_zero[1]:+.4f}, {a_zero[2]:+.4f})  diff={d_zero:.4f}")

    np.random.seed(42)
    random_state = (np.random.randn(15) * 0.5).tolist()
    random_state[7] = 0.08
    a_rand = query(front, wrist, random_state)
    d_rand = np.linalg.norm(a_rand[:3] - a_real[:3])
    print(f"  Random state → EE_delta=({a_rand[0]:+.4f}, {a_rand[1]:+.4f}, {a_rand[2]:+.4f})  diff={d_rand:.4f}")

    if d_zero < 0.001 and d_rand < 0.001:
        print("CONCLUSION: Model IGNORES state (vision-dominated)")
    elif d_zero > 0.01:
        print("CONCLUSION: Model heavily depends on STATE")
    else:
        print(f"CONCLUSION: State has SOME influence (diff={d_zero:.4f})")

    # Save the images for inspection
    for label, (cx, cy) in positions.items():
        set_cube(env, cx, cy)
        obs = env._get_obs()
        Image.fromarray(obs["observation.images.front"]).save(f"cube_{label.strip()}.png")
    print("\nSaved cube_left.png, cube_center.png, cube_right.png for inspection")

    env.close()

if __name__ == "__main__":
    main()
