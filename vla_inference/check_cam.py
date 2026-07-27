"""Quick camera view check — renders one frame and saves as PNG."""
import sys, numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from vla_inference.env.panda_joint_env import PandaJointEnv

env = PandaJointEnv(render_mode="rgb_array", scene="task")
obs = env.reset()

# Reach toward cubes using EE control (10 intermediate steps, 10 final steps)
labels = ["begin", "mid", "end"]

# Begin: home position (already rendered at reset)
Image.fromarray(obs["observation.images.front"]).save(f"cam_front_begin.png")
Image.fromarray(obs["observation.images.wrist"]).save(f"cam_wrist_begin.png")
print("Saved begin")

# Mid: move toward cubes over 20 steps
for _ in range(20):
    obs = env.step_ee(np.array([0.02, 0.01, -0.015]), 0.3)
Image.fromarray(obs["observation.images.front"]).save("cam_front_mid.png")
Image.fromarray(obs["observation.images.wrist"]).save("cam_wrist_mid.png")
print("Saved mid")

# End: lower down + close gripper over 20 steps
for _ in range(20):
    obs = env.step_ee(np.array([0.01, 0.005, -0.025]), 0.8)
Image.fromarray(obs["observation.images.front"]).save("cam_front_end.png")
Image.fromarray(obs["observation.images.wrist"]).save("cam_wrist_end.png")
print("Saved end")

env.close()
