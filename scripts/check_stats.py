import json, numpy as np
with open('/root/autodl-tmp/my_dataset/meta/stats.json') as f:
    stats = json.load(f)
action = stats['action']
print('Action per-dim stats:')
names = ['dx','dy','dz','drx','dry','drz','grip']
for i, name in enumerate(names):
    m = action['mean'][i]
    s = action['std'][i]
    mn = action['min'][i]
    mx = action['max'][i]
    rmse = np.sqrt(0.1 / 7) * s
    print(f'  {name}: mean={m:+.4f}  std={s:.4f}  min={mn:+.4f}  max={mx:+.4f}  est_RMSE={rmse:.4f}')
