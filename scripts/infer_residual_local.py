"""Local MuJoCo inference — server handles residual correction.

Setup:
    1. Server: python policy_server.py --checkpoint <path> --residual-model residual_net.pt
    2. Local:  ssh -p 27074 -L 8765:localhost:8765 root@connect.westd.seetacloud.com
    3. Local:  python scripts/infer_residual_local.py --episodes 5
"""

import argparse, base64, io, json, sys, time, urllib.request, urllib.error
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

SERVER_URL = "http://localhost:8765/predict"


def encode_image(img_array: np.ndarray) -> str:
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def query_policy(obs: dict, task: str = "move to the red cube and pick it up") -> dict:
    """Send observation to server, get action (+ corrected_action if residual loaded)."""
    payload = json.dumps({
        "front_image": encode_image(obs["observation.images.front"]),
        "wrist_image": encode_image(obs["observation.images.wrist"]),
        "state": obs["observation.state"].tolist(),
        "task": task,
    }).encode()

    req = urllib.request.Request(SERVER_URL, data=payload,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Server error:\n{e.read().decode()}") from e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()

    print(f"Server: {SERVER_URL}")

    from vla_inference.env.panda_joint_env import PandaJointEnv
    env = PandaJointEnv(render_mode="human")
    print("MuJoCo env ready.\n")

    for ep in range(args.episodes):
        obs = env.reset()
        step = 0
        total_lat = 0.0

        print(f"{'='*60}")
        print(f"Episode {ep+1}/{args.episodes}")
        print(f"{'Step':>5} {'Lat(ms)':>7} {'EE(x,y,z)':>22} {'Act(dx,dy,dz)':>22} {'Δ(dx,dy,dz)':>22}")
        print(f"{'─'*5} {'─'*7} {'─'*22} {'─'*22} {'─'*22}")

        while step < args.max_steps:
            t0 = time.perf_counter()

            result = query_policy(obs)

            # Use corrected action if server has residual model
            if "corrected_action" in result:
                action = np.array(result["corrected_action"])
                bc = np.array(result["action"])
                delta = action[:3] - bc[:3]
            else:
                action = np.array(result["action"])
                delta = np.zeros(3)

            lat = (time.perf_counter() - t0) * 1000
            total_lat += lat

            if step < 15 or step % 10 == 0:
                ee = env.ee_position
                print(f"{step:5d} {lat:7.0f}  "
                      f"[{ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}]         "
                      f"({action[0]:+.3f},{action[1]:+.3f},{action[2]:+.3f})         "
                      f"({delta[0]:+.4f},{delta[1]:+.4f},{delta[2]:+.4f})")

            obs = env.apply_ee_delta(action)
            step += 1

        avg_lat = total_lat / max(step, 1)
        ee = env.ee_position
        print(f"  Done | final EE:[{ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}] | avg_lat={avg_lat:.0f}ms")

        time.sleep(1.0)

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
