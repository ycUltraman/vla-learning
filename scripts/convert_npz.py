import numpy as np, shutil, random, json
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SRC = Path("/root/autodl-tmp/collected_episodes")
DST = Path("/root/autodl-tmp/my_dataset")
TASK = "move to the red cube and pick it up"
FPS = 10

if DST.exists():
    shutil.rmtree(str(DST))

dataset = LeRobotDataset.create(
    repo_id="my_dataset", root=str(DST), fps=FPS,
    features={
        "observation.state": {"dtype": "float32", "shape": (15,), "names": ["j1","j2","j3","j4","j5","j6","j7","grip_w","ee_x","ee_y","ee_z","qw","qx","qy","qz"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["dx","dy","dz","drx","dry","drz","gripper"]},
        "observation.images.front": {"dtype": "video", "shape": (480, 640, 3), "names": ["height","width","channel"]},
        "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3), "names": ["height","width","channel"]},
    },
)

eps = sorted(SRC.glob("ep_*"))
random.seed(42)
random.shuffle(eps)

n_train = int(len(eps) * 0.9)
print(f"Total: {len(eps)} episodes, train: {n_train}, val: {len(eps) - n_train}")

for ep in eps:
    traj = np.load(ep / "trajectory.npz", allow_pickle=True)
    front_npz = np.load(ep / "images_front.npz", allow_pickle=True)
    wrist_npz = np.load(ep / "images_wrist.npz", allow_pickle=True)

    states = traj["states"].astype(np.float32)
    actions = traj["actions"].astype(np.float32)
    front = front_npz[front_npz.files[0]].astype(np.uint8)
    wrist = wrist_npz[wrist_npz.files[0]].astype(np.uint8)

    if front.shape[1] == 3: front = np.transpose(front, (0, 2, 3, 1))
    if wrist.shape[1] == 3: wrist = np.transpose(wrist, (0, 2, 3, 1))

    n = min(len(states), len(actions), len(front), len(wrist))
    print(f"{ep.name} s{len(states)} a{len(actions)} f{len(front)} w{len(wrist)} -> using {n}")

    for i in range(n):
        dataset.add_frame({
            "observation.state": states[i],
            "action": actions[i],
            "observation.images.front": front[i],
            "observation.images.wrist": wrist[i],
            "task": TASK,
        })
    dataset.save_episode()
    print(f"  saved -> total_eps={dataset.meta.total_episodes} total_frames={dataset.meta.total_frames}")

print(f"\nDONE: {len(eps)} eps, {dataset.meta.total_frames} frames -> {DST}")
print(f"Shuffled order — use eval_split in train_config for random train/val split")
