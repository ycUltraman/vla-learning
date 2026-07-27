"""Convert collected NPZ+MP4 data to LeRobot v3.0 format for PI0.5 training.

Usage (on server):
    python convert_to_lerobot.py --input collected_data_lerobot --output teleop_dataset
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="collected_data_lerobot")
    parser.add_argument("--output", default="teleop_dataset")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = sorted(input_dir.glob("episode_*/"))
    if not episodes:
        print(f"No episodes found in {input_dir}")
        return

    print(f"Found {len(episodes)} episodes")
    total_frames = 0
    episode_metas = []
    task_name = "pick up the red cube and place it on the pad"

    # ── 1. Write data parquet ──
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    all_rows = []
    ep_lengths = []

    for ep_idx, ep_dir in enumerate(episodes):
        data = np.load(ep_dir / "data.npz")
        states = data["states"]  # (T, 15)
        actions = data["actions"]  # (T, 8)
        n_frames = len(states)
        ep_lengths.append(n_frames)

        for frame_idx in range(n_frames):
            row = {
                "observation.state": states[frame_idx].tolist(),
                "action": actions[frame_idx].tolist(),
                "timestamp": frame_idx / args.fps,
                "frame_index": frame_idx,
                "episode_index": ep_idx,
                "index": total_frames + frame_idx,
                "task_index": 0,
                "task": task_name,
            }
            all_rows.append(row)

        total_frames += n_frames
        episode_metas.append({"episode_index": ep_idx, "tasks": [task_name]})
        print(f"  Ep {ep_idx}: {n_frames} frames")

    # Write single parquet
    data_dir = output_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, data_dir / "file-000.parquet")
    print(f"\nTotal: {total_frames} frames in {len(episodes)} episodes")

    # ── 2. Merge videos ──
    for cam in ("front", "wrist"):
        video_dir = output_dir / "videos" / f"observation.images.{cam}" / "chunk-000"
        video_dir.mkdir(parents=True, exist_ok=True)

        # Concatenate all episode videos using ffmpeg concat
        concat_list = output_dir / f"_{cam}_concat.txt"
        with open(concat_list, "w") as f:
            for ep_dir in episodes:
                mp4 = ep_dir / f"{cam}.mp4"
                if mp4.exists():
                    f.write(f"file '{mp4.resolve()}'\n")
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(video_dir / "file-000.mp4"),
        ], check=True)
        concat_list.unlink()
        print(f"  Merged {cam} videos")

    # ── 3. Write meta ──
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(exist_ok=True)

    info = {
        "codebase_version": "v3.0",
        "robot_type": "franka_panda",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": 1,
        "fps": args.fps,
        "splits": ["train"],
        "features": {
            "observation.state": {"dtype": "float32", "shape": [15], "names": None},
            "action": {"dtype": "float32", "shape": [8], "names": None},
            "observation.images.front": {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channel"]},
            "observation.images.wrist": {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channel"]},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    import pandas as pd
    tasks_df = pd.DataFrame({"task_index": [0], "task": [task_name]})
    tasks_df.set_index("task", inplace=True)
    tasks_df.to_parquet(meta_dir / "tasks.parquet")

    # Episodes stats
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep_idx, meta in enumerate(episode_metas):
            meta["length"] = ep_lengths[ep_idx]
            f.write(json.dumps(meta) + "\n")

    print(f"\nLeRobot dataset created at {output_dir}/")
    print(f"Ready for training: lerobot-train --dataset.repo_id=<name> --root={output_dir}")


if __name__ == "__main__":
    main()
