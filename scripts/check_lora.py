"""Check which modules have LoRA applied — run on server."""
import json

with open('/root/autodl-tmp/output_my_data/checkpoints/004000/pretrained_model/adapter_config.json') as f:
    cfg = json.load(f)

print(f"LoRA r={cfg['r']}, alpha={cfg['lora_alpha']}")
print(f"\ntarget_modules:")
print(cfg['target_modules'])
print()

# Parse what's covered
tm = cfg['target_modules']
if 'gemma_expert' in tm:
    print("✅ Gemma expert (text LLM) — Q/V attention projection have LoRA")
if 'state_proj' in tm:
    print("✅ Action head state projection — LoRA")
if 'action_in_proj' in tm:
    print("✅ Action head input projection — LoRA")
if 'action_out_proj' in tm:
    print("✅ Action head output projection — LoRA")
if 'action_time_mlp' in tm:
    print("✅ Action head time MLP — LoRA")
if 'vision' not in tm.lower():
    print("\n❌ Vision encoder: NO LoRA (fully frozen)")
if 'paligemma' not in tm.lower() and 'language' not in tm.lower():
    print("❌ PaliGemma language backbone: NO LoRA (frozen except gemma_expert q/v)")
