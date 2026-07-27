"""Test whether PI0.5 relies on state vs image for action prediction.

Test 1: Same image, random state → does action change? (state sensitivity)
Test 2: Same state, black image → does action change? (image sensitivity)
"""

import json, urllib.request, urllib.error
import numpy as np
from pathlib import Path

SERVER = "http://localhost:8765/predict"

def query(front_b64, wrist_b64, state_list):
    payload = json.dumps({
        "front_image": front_b64,
        "wrist_image": wrist_b64,
        "state": state_list,
        "task": "pick up the red cube",
    }).encode()
    req = urllib.request.Request(SERVER, data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return np.array(json.loads(resp.read())["action"])


def main():
    import sys, base64, io
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from vla_inference.env.panda_joint_env import PandaJointEnv
    from PIL import Image

    env = PandaJointEnv(render_mode="rgb_array", scene="task")
    obs = env.reset()
    # Move to a mid-reach position
    for _ in range(10):
        obs = env.step_ee(np.array([0.01, 0.005, -0.01]), 0.3)

    def encode(img):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    front_b64 = encode(obs["observation.images.front"])
    wrist_b64 = encode(obs["observation.images.wrist"])
    real_state = obs["observation.state"].tolist()
    black_b64 = encode(np.zeros((480, 640, 3), dtype=np.uint8))

    # Test 1: Same image, different states
    print("=" * 60)
    print("Test 1: Same image, different states")
    print("=" * 60)
    a_real = query(front_b64, wrist_b64, real_state)
    print(f"  Real state   → action[:7] = {a_real[:7].round(3)}")

    # Random states (within training range)
    for label, fake_state in [
        ("Home   ", [0,0,0,-1.57,0,1.57,-0.78,0.08,0.55,0,0.47,0,0.707,0.707,0]),
        ("Low arm", [0.2,-0.3,0.1,-1.8,0,1.8,-0.8,0.06,0.44,0.05,0.1,0,0.707,0.707,0]),
        ("High arm", [0,0,0,-1.0,0,0.5,-0.5,0.04,0.3,-0.1,0.6,0,0.707,0.707,0]),
        ("Grip cls", [0,0,0,-1.57,0,1.57,-0.78,0.0,0.55,0,0.47,0,0.707,0.707,0]),
    ]:
        a = query(front_b64, wrist_b64, fake_state)
        diff = np.linalg.norm(a[:7] - a_real[:7])
        print(f"  {label} → action[:7] = {a[:7].round(3)}  diff={diff:.3f}")

    # Test 2: Same state, black image
    print()
    print("=" * 60)
    print("Test 2: Same state, black image")
    print("=" * 60)
    a_black = query(black_b64, black_b64, real_state)
    diff_black = np.linalg.norm(a_black[:7] - a_real[:7])
    print(f"  Real image  → action[:7] = {a_real[:7].round(3)}")
    print(f"  Black image → action[:7] = {a_black[:7].round(3)}  diff={diff_black:.3f}")

    print()
    if diff_black > 0.2:
        print("CONCLUSION: Model relies heavily on IMAGE (big change with black image)")
    else:
        print("CONCLUSION: Model relies mainly on STATE")
    env.close()

if __name__ == "__main__":
    main()
