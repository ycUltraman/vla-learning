import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path('/root/autodl-tmp/collected_episodes')
DST = Path('/root/autodl-tmp/my_dataset')
FPS = 10
TASK = 'pick the red cube and place it on the pad'

ep_dirs = sorted(d for d in SRC.iterdir() if d.is_dir() and d.name.startswith('ep_'))
N = len(ep_dirs)
print(f'{N} episodes')

chunk = 'chunk-000'
for d in [DST/'data'/chunk, DST/'videos'/'observation.images.front'/chunk,
          DST/'videos'/'observation.images.wrist'/chunk, DST/'meta']:
    d.mkdir(parents=True, exist_ok=True)

episodes_meta = []
total_frames = 0

for i, ep_dir in enumerate(ep_dirs):
    print(f'Ep {i}/{N} ...', end='', flush=True)
    traj = np.load(ep_dir / 'trajectory.npz')
    states = traj['states']
    actions = traj['actions']
    n = len(states)
    total_frames += n

    for cam in ['front', 'wrist']:
        frames = np.load(ep_dir / f'images_{cam}.npz')['frames']
        h, w = frames.shape[1:3]
        mp4_path = str(DST / 'videos' / f'observation.images.{cam}' / chunk / f'file-{i:03d}.mp4')
        proc = subprocess.Popen(['ffmpeg', '-y', '-v', 'quiet',
            '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{w}x{h}',
            '-pix_fmt', 'rgb24', '-r', str(FPS), '-i', '-',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-preset', 'fast',
            mp4_path], stdin=subprocess.PIPE)
        for f in frames:
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        proc.wait()

    rows = [{
        'observation.state': states[j].tolist(),
        'action': actions[j].tolist(),
        'timestamp': j / FPS,
        'frame_index': j,
        'episode_index': i,
        'index': i * 100000 + j,
        'task_index': 0,
        'task': TASK,
    } for j in range(n)]
    pd.DataFrame(rows).to_parquet(DST / 'data' / chunk / f'episode_{i:06d}.parquet', index=False)
    episodes_meta.append({'episode_index': i, 'tasks': [TASK], 'length': n})
    print(f' {n} frames')

info = {
    'codebase_version': 'v3.0', 'robot_type': 'franka_panda',
    'total_episodes': N, 'total_frames': total_frames, 'total_tasks': 1,
    'total_chunks': 1, 'total_videos': 2 * N, 'chunks_size': 1000, 'fps': FPS,
    'splits': ['train'],
    'data_path': 'data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet',
    'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
    'features': {
        'observation.state': {'dtype': 'float32', 'shape': [15]},
        'action': {'dtype': 'float32', 'shape': [8]},
        'observation.images.front': {'dtype': 'video', 'shape': [480, 640, 3]},
        'observation.images.wrist': {'dtype': 'video', 'shape': [480, 640, 3]},
    },
}
json.dump(info, open(DST/'meta'/'info.json', 'w'), indent=2)
with open(DST/'meta'/'episodes.jsonl', 'w') as f:
    for ep in episodes_meta:
        f.write(json.dumps(ep) + '\n')
with open(DST/'meta'/'tasks.jsonl', 'w') as f:
    f.write(json.dumps({'task_index': 0, 'task': TASK}) + '\n')
json.dump({
    'observation.state': {'mean': [0.0]*15, 'std': [1.0]*15},
    'action': {'mean': [0.0]*8, 'std': [1.0]*8},
}, open(DST/'meta'/'stats.json', 'w'), indent=2)
print(f'Done: {N} episodes, {total_frames} frames')
