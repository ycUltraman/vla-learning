"""Remote PI0.5 inference with local MuJoCo rendering.

Architecture:
    [Local MuJoCo] --HTTP/SSH tunnel--> [AutoDL PI0.5 policy server]

The server runs `policy_server.py` on the AutoDL machine.
Connect via SSH tunnel before running this script:
    ssh -p 12956 -L 8765:localhost:8765 root@connect.westd.seetacloud.com

Then:
    python vla_inference/infer.py
"""

import argparse
import base64
import io
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from vla_inference.env.panda_joint_env import PandaJointEnv

DEFAULT_SERVER = "http://localhost:8765/predict"
DEFAULT_TASK = "move to the red cube and pick it up"


def encode_image(img_array: np.ndarray) -> str:
    """Encode (H, W, 3) uint8 numpy array → base64 JPEG string."""
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def query_policy(server_url: str, obs: dict, task: str = DEFAULT_TASK) -> np.ndarray:
    """Send observation to policy server, return (8,) action."""
    payload = json.dumps({
        "front_image": encode_image(obs["observation.images.front"]),
        "wrist_image": encode_image(obs["observation.images.wrist"]),
        "state": obs["observation.state"].tolist(),
        "task": task,
    }).encode()

    req = urllib.request.Request(
        server_url, data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
        return np.array(result["action"])
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"Server error:\n{body}") from e


def main():
    parser = argparse.ArgumentParser(description="PI0.5 MuJoCo remote inference")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Policy server URL")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    print(f"Policy server: {args.server}")
    print("Creating MuJoCo environment ...")
    env = PandaJointEnv(render_mode="human")
    obs = env.reset()

    print(f"State dim: {obs['observation.state'].shape[0]}")
    print(f"Front image: {obs['observation.images.front'].shape}")
    print(f"Wrist image: {obs['observation.images.wrist'].shape}")
    labels = ['j1','j2','j3','j4','j5','j6','j7','grip_w','ee_x','ee_y','ee_z','ee_qw','ee_qx','ee_qy','ee_qz']
    print("Initial state:")
    for j in range(15):
        print(f"  {labels[j]:6s} = {obs['observation.state'][j]:+.4f}")
    print("\nStarting inference loop. Close viewer to stop.\n")

    try:
        for step in range(args.steps):
            t0 = time.perf_counter()

            action = query_policy(args.server, obs)

            # Safety checks
            if np.any(np.isnan(action)):
                raise RuntimeError(f"NaN action at step {step}: {action}")
            if np.max(np.abs(action[:6])) > 0.5:  # EE delta > 50cm is unsafe
                raise RuntimeError(f"Action too large at step {step}: {action[:7]}")

            obs = env.apply_ee_delta(action)

            latency = (time.perf_counter() - t0) * 1000

            if step < 30 or step % 10 == 0:
                ee = env.ee_position
                action_str = " ".join(f"{a:+.3f}" for a in action)
                print(
                    f"Step {step:4d} | Lat: {latency:6.0f}ms | "
                    f"EE: [{ee[0]:.3f} {ee[1]:.3f} {ee[2]:.3f}] | "
                    f"Grip: {env.gripper_width:.3f}"
                )
                print(f"         action: [{action_str}]")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()
