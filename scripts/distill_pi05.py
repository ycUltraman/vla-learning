"""Distill PI0.5 → state-only MLP: run BC, collect (state, action), train MLP.

The MLP learns to mimic PI0.5 with access to cube position (from env).
This gives the state-only MLP "cube awareness" for PPO initialization.

Usage (on server):
    source ~/autodl-tmp/venv_lerobot/bin/activate
    export HF_HOME=/root/autodl-tmp/.hf_cache HF_HUB_OFFLINE=1
    MUJOCO_GL=egl python distill_pi05.py \
        --checkpoint <path> --collect_eps 200 --epochs 50
"""

import argparse, warnings
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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
    parser.add_argument("--collect_eps", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save", default="./distilled_policy.pt")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Device: {device}")

    # Load PI0.5
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.configs import PreTrainedConfig
    from peft import PeftConfig, PeftModel
    print(f"Loading PI0.5 from {args.checkpoint} ...")
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.pretrained_path = args.checkpoint
    peft_cfg = PeftConfig.from_pretrained(args.checkpoint)
    base = PI05Policy.from_pretrained(peft_cfg.base_model_name_or_path, config=cfg)
    pi05 = PeftModel.from_pretrained(base, args.checkpoint, config=peft_cfg)
    pi05 = pi05.merge_and_unload()
    pi05 = pi05.to(device=device, dtype=torch.float32)
    pi05.eval()
    pre, post = make_pre_post_processors(policy_cfg=pi05.config, pretrained_path=args.checkpoint)
    print("PI0.5 loaded.")

    env = PandaRLEnv()

    # Collect
    print(f"\nCollecting {args.collect_eps} episodes from PI0.5 ...")
    all_states, all_actions = [], []
    for ep in range(args.collect_eps):
        env.reset()
        obs = env.get_obs_pi05()
        done = False
        while not done:
            batch = pre(build_batch(obs, device))
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = pi05.select_action(batch)
                act = post(raw).squeeze(0).cpu().numpy()
            state_14 = np.concatenate([
                obs["observation.state"][:7],   # joints
                obs["observation.state"][8:11], # ee
                env.cube_position.copy(),        # cube!
                [obs["observation.state"][7]],   # grip
            ]).astype(np.float32)
            act_4 = np.array([act[0], act[1], act[2], act[6]], dtype=np.float32)
            all_states.append(state_14)
            all_actions.append(act_4)
            rl = np.array([act[0], act[1], act[2], act[6]])
            _, _, t, tr, _ = env.step(rl)
            done = t or tr
            obs = env.get_obs_pi05()
        if ep % 20 == 0 or ep < 5:
            print(f"  ep {ep:4d}: {env._step_count:3d} steps")

    X = np.array(all_states)
    Y = np.array(all_actions)
    print(f"\nCollected {len(X)} frames, state_dim={X.shape[1]}, action_dim={Y.shape[1]}")

    # Train
    x_mean, x_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    y_mean, y_std = Y.mean(axis=0), Y.std(axis=0) + 1e-8
    X_n = (X - x_mean) / x_std
    Y_n = (Y - y_mean) / y_std

    dataset = TensorDataset(torch.tensor(X_n, dtype=torch.float32),
                            torch.tensor(Y_n, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = nn.Sequential(
        nn.Linear(14, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 4),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\nTraining {args.epochs} epochs ...")
    for epoch in range(args.epochs):
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx)
            loss = nn.functional.mse_loss(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}: loss={total_loss/len(loader):.4f}")

    torch.save({"model_state": model.state_dict(),
                "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std}, args.save)
    print(f"Saved: {args.save}")
    env.close()


if __name__ == "__main__":
    main()
