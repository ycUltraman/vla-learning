"""Plot action distribution histograms from LeRobot dataset.
Usage: python check_actions.py
"""
import numpy as np
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("my_dataset", root="E:/Verilog/vla-learning/collected_episodes")  # adjust path
n = ds.meta.total_frames
print(f"Loading {n} frames...")

all_a = []
for i in range(n):
    s = ds[i]
    a = s["action"]
    if hasattr(a, "numpy"): a = a.numpy()
    all_a.append(np.array(a).flatten())
actions = np.array(all_a)

labels = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7","gripper"]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for j in range(8):
    ax = axes[j]
    ax.hist(actions[:, j], bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    ax.set_title(labels[j])
    ax.set_xlabel('rad' if j < 7 else 'cmd')
    ax.axvline(actions[:, j].mean(), color='red', linestyle='--')
fig.tight_layout()
plt.savefig('action_hist.png', dpi=100)
print("Saved action_hist.png")
