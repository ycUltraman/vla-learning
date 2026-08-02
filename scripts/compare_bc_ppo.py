"""AB test: BC vs PPO in the SAME env with the SAME code path."""
import sys, time, warnings
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from panda_rl_env import PandaRLEnv

def main():
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.configs import PreTrainedConfig
    from peft import PeftConfig, PeftModel

    device = torch.device("cuda")
    ckpt = "/root/autodl-tmp/output_my_data/checkpoints/020000/pretrained_model"
    print(f"Loading PI0.5 from {ckpt} ...")
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    peft_config = PeftConfig.from_pretrained(ckpt)
    base = PI05Policy.from_pretrained(peft_config.base_model_name_or_path, config=cfg)
    pi05 = PeftModel.from_pretrained(base, ckpt, config=peft_config)
    pi05 = pi05.merge_and_unload()
    pi05 = pi05.to(device=device, dtype=torch.float32)
    pi05.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=pi05.config, pretrained_path=ckpt)
    print("Loaded.")

    def build_batch(obs_dict):
        front = torch.from_numpy(obs_dict["observation.images.front"]).float() / 255.0
        wrist = torch.from_numpy(obs_dict["observation.images.wrist"]).float() / 255.0
        state = torch.from_numpy(obs_dict["observation.state"]).float()
        return {
            "observation.images.front": front.permute(2, 0, 1).unsqueeze(0).to(device),
            "observation.images.wrist": wrist.permute(2, 0, 1).unsqueeze(0).to(device),
            "observation.state": state.unsqueeze(0).to(device),
            "task": "move to the red cube and pick it up",
        }

    for mode in ["BC", "PPO"]:
        env = PandaRLEnv()
        success = 0
        for ep in range(20):
            env.reset()
            obs = env.get_obs_pi05()
            done = False
            while not done:
                batch = preprocessor(build_batch(obs))
                with torch.no_grad(), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if mode == "BC":
                        raw = pi05.select_action(batch)
                        action = postprocessor(raw).squeeze(0).cpu().numpy()
                    else:  # PPO mode: simulate what PPO does (residual=0 initially)
                        raw = pi05.select_action(batch)
                        bc = postprocessor(raw).squeeze(0).cpu().numpy()
                        action = bc.copy()  # residual=0 means PPO=BC

                rl_action = np.array([action[0], action[1], action[2], action[6]])
                _, _, terminated, truncated, _ = env.step(rl_action)
                done = terminated or truncated
                obs = env.get_obs_pi05()

            if env._success:
                success += 1
            if ep < 5:
                print(f"  [{mode}] ep{ep}: steps={env._step_count} success={env._success}")
        print(f"\n{mode}: {success}/20 = {success*5:.0f}%")
        env.close()

if __name__ == "__main__":
    main()
