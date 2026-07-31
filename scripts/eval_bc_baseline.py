"""Evaluate pure BC (deterministic, no PPO) on the pick task.

Usage (on server):
    source ~/autodl-tmp/venv_lerobot/bin/activate
    export HF_HOME=/root/autodl-tmp/.hf_cache HF_HUB_OFFLINE=1
    MUJOCO_GL=egl python eval_bc_baseline.py \
        --checkpoint /root/autodl-tmp/output_my_data/checkpoints/020000/pretrained_model \
        --episodes 100
"""

import argparse, time, warnings
import numpy as np
import torch
from panda_rl_env import PandaRLEnv


def build_batch(obs_dict, device):
    front = torch.from_numpy(obs_dict["observation.images.front"]).float() / 255.0
    wrist = torch.from_numpy(obs_dict["observation.images.wrist"]).float() / 255.0
    state = torch.from_numpy(obs_dict["observation.state"]).float()
    return {
        "observation.images.front": front.permute(2, 0, 1).unsqueeze(0).to(device),
        "observation.images.wrist": wrist.permute(2, 0, 1).unsqueeze(0).to(device),
        "observation.state": state.unsqueeze(0).to(device),
        "task": "move to the red cube and pick it up",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Device: {device}")

    # Load PI0.5 (same as train_ppo_pi05.py)
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.configs import PreTrainedConfig
    from peft import PeftConfig, PeftModel

    print(f"Loading PI0.5 from {args.checkpoint} ...")
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.pretrained_path = args.checkpoint
    peft_config = PeftConfig.from_pretrained(args.checkpoint)
    base_policy = PI05Policy.from_pretrained(
        peft_config.base_model_name_or_path, config=cfg)
    pi05 = PeftModel.from_pretrained(base_policy, args.checkpoint, config=peft_config)
    pi05 = pi05.merge_and_unload()
    pi05 = pi05.to(device=device, dtype=torch.float32)
    pi05.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=pi05.config, pretrained_path=args.checkpoint)
    print("PI0.5 loaded.")

    env = PandaRLEnv()
    results = {"success": 0, "steps": [], "final_xy_err": [], "grasped": 0}

    t_start = time.time()
    for ep in range(args.episodes):
        env.reset()
        obs = env.get_obs_pi05()
        done = False

        while not done:
            batch = preprocessor(build_batch(obs, device))
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = pi05.select_action(batch)
                action = postprocessor(raw).squeeze(0).cpu().numpy()

            rl_action = np.array([action[0], action[1], action[2], action[6]])
            _, _, terminated, truncated, _ = env.step(rl_action)
            done = terminated or truncated
            obs = env.get_obs_pi05()

        results["steps"].append(env._step_count)
        cube, ee = env.cube_position, env.ee_position
        xy_err = np.linalg.norm(ee[:2] - cube[:2])
        results["final_xy_err"].append(xy_err)
        if env._success:
            results["success"] += 1
        if env._grasped:
            results["grasped"] += 1

        if ep % 10 == 0 or ep < 5:
            print(f"Ep {ep:3d} | steps: {env._step_count:3d} | "
                  f"xy_err: {xy_err:.3f} | grasped: {env._grasped} | success: {env._success}")

    total_t = time.time() - t_start
    n = args.episodes
    print(f"\n{'='*50}")
    print(f"BC Baseline ({n} episodes, {total_t/60:.1f}min)")
    print(f"  Success rate:   {results['success']}/{n} = {results['success']/n*100:.1f}%")
    print(f"  Grasp rate:     {results['grasped']}/{n} = {results['grasped']/n*100:.1f}%")
    print(f"  Avg steps:      {np.mean(results['steps']):.0f}")
    print(f"  Median steps:   {np.median(results['steps']):.0f}")
    print(f"  Final XY err:   mean={np.mean(results['final_xy_err']):.3f}m  median={np.median(results['final_xy_err']):.3f}m")
    env.close()


if __name__ == "__main__":
    main()
