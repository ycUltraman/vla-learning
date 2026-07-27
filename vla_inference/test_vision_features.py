"""Test if vision encoder distinguishes left vs right cube positions."""

import base64, io, json, sys, urllib.request
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from vla_inference.env.panda_joint_env import PandaJointEnv
import mujoco

SERVER = "http://localhost:8765/predict"

def query_features(front_b64, wrist_b64, state_list):
    """Query policy server for vision features."""
    payload = json.dumps({
        "front_image": front_b64, "wrist_image": wrist_b64,
        "state": state_list, "task": "move to the red cube",
        "return_features": True,
    }).encode()
    req = urllib.request.Request(SERVER, data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())

def to_b64(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def set_cube(env, x, y, z=0.03):
    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
    adr = env.model.jnt_qposadr[env.model.body_jntadr[body_id]]
    env.data.qpos[adr:adr+3] = [x, y, z]
    mujoco.mj_forward(env.model, env.data)

def main():
    env = PandaJointEnv(render_mode="rgb_array", scene="task")
    obs = env.reset()
    home_state = obs["observation.state"].tolist()

    positions = {"far_left": (0.45, -0.40), "far_right": (0.45, 0.40)}

    for label, (cx, cy) in positions.items():
        set_cube(env, cx, cy)
        front = to_b64(env._get_obs()["observation.images.front"])
        wrist = to_b64(env._get_obs()["observation.images.wrist"])
        result = query_features(front, wrist, home_state)
        print(f"{label}: {json.dumps(result, default=str)[:300]}")

    env.close()

if __name__ == "__main__":
    main()
