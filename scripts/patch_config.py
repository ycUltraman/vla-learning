import sys
path = sys.argv[1]
with open(path) as f:
    content = f.read()

# 1. r=32 -> r=64, alpha=32 -> alpha=64
content = content.replace('  r: 32\n', '  r: 64\n')
content = content.replace('  lora_alpha: 32\n', '  lora_alpha: 64\n')

# 2. steps=20000 -> 40000
content = content.replace('steps: 20000\n', 'steps: 40000\n')

# 3. Add optimizer_lr and scheduler_decay_lr
# Insert after batch_size line
old = 'batch_size: 4\nnum_workers: 16\n'
new = ('batch_size: 4\nnum_workers: 16\n'
       'optimizer_lr: 1.0e-4\n'
       'scheduler_decay_lr: 1.0e-5\n'
       'scheduler_decay_steps: 40000\n')
content = content.replace(old, new, 1)

with open(path, 'w') as f:
    f.write(content)

# Print changed sections
for i, line in enumerate(content.split('\n')):
    if any(kw in line.lower() for kw in ['peft', 'r:', 'lora', 'step', 'optim', 'sched', 'batch']):
        print(f'  {line}')
