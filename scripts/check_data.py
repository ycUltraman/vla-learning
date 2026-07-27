"""Quick data check — verify parquets + videos are valid."""
import pandas as pd, json, subprocess, sys
from pathlib import Path

d = Path("/root/autodl-tmp/my_dataset")

parquets = sorted((d / "data" / "chunk-000").glob("*.parquet"))
print(f"Parquets: {len(parquets)}")
df = pd.read_parquet(parquets[0])
print(f"Columns: {list(df.columns)}")
print(f"First state: {len(df.iloc[0]['observation.state'])}D")
print(f"First action: {len(df.iloc[0]['action'])}D")

front_vids = list((d / "videos" / "observation.images.front" / "chunk-000").glob("*.mp4"))
wrist_vids = list((d / "videos" / "observation.images.wrist" / "chunk-000").glob("*.mp4"))
print(f"Videos: front={len(front_vids)} wrist={len(wrist_vids)}")

for v in front_vids[:1]:
    result = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries",
        "stream=width,height,nb_frames", "-of", "csv=p=0", str(v)],
        capture_output=True, text=True)
    print(f"Front video: {result.stdout.strip()}")

with open(d / "meta" / "info.json") as f:
    info = json.load(f)
print(f"Meta: {info['total_episodes']} eps, {info['total_frames']} frames")

print("ALL GOOD")
