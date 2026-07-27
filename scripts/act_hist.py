"""Action histogram — run on server. Uses parquet directly (fast)."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_parquet('/root/autodl-tmp/my_dataset_v3/data/chunk-000/file-000.parquet')
actions = np.array(df['action'].apply(list).tolist(), dtype=np.float32)
print(f"Loaded {len(actions)} frames")

labels = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "gripper"]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for j in range(8):
    ax = axes[j]
    ax.hist(actions[:, j], bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    ax.set_title(labels[j])
    ax.set_xlabel('rad' if j < 7 else 'cmd')
    ax.axvline(actions[:, j].mean(), color='red', linestyle='--', linewidth=1)
fig.tight_layout()
plt.savefig('/root/autodl-tmp/action_hist.png', dpi=100)
print("Saved /root/autodl-tmp/action_hist.png")
