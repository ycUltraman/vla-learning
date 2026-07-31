"""Train state-only MLP via Behavior Cloning on teleop data.

Maps recorded 15D states -> 4D actions. The MLP learns to mimic
human demonstrations without vision. This becomes the PPO starting point.

State mapping: 15D [j1..j7, grip, ee_xyz, quat_wxyz]
              -> 14D [j1..j7, ee_xyz, cube_xyz=0, grip]
Action: 7D [dx, dy, dz, drx, dry, drz, grip] -> 4D [dx, dy, dz, grip]

Usage:
    python scripts/train_bc.py --data collected_episodes/ --epochs 50
"""

import argparse, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_all_episodes(data_dir):
    """Load all episodes, return (states_14d, actions_4d)."""
    data_path = Path(data_dir)
    all_states, all_actions = [], []
    for ep_dir in sorted(data_path.glob("ep_*")):
        traj = np.load(ep_dir / "trajectory.npz", allow_pickle=True)
        states_15 = traj["states"].astype(np.float32)   # (N, 15)
        actions_7 = traj["actions"].astype(np.float32)   # (N, 7)

        # 15D -> 14D: drop quat[4], add cube=0[3], reorder
        # 15D = [j1..j7(7), grip_w(1), ee_xyz(3), quat_wxyz(4)]
        # 14D = [j1..j7(7), ee_xyz(3), cube_xyz(3), grip_w(1)]
        joints = states_15[:, :7]          # (N, 7)
        grip = states_15[:, 7:8]           # (N, 1)
        ee = states_15[:, 8:11]            # (N, 3)
        cube = np.zeros((len(states_15), 3), dtype=np.float32)  # placeholder
        states_14 = np.concatenate([joints, ee, cube, grip], axis=1)

        # 7D -> 4D: drop drx, dry, drz
        actions_4 = np.stack([
            actions_7[:, 0], actions_7[:, 1], actions_7[:, 2], actions_7[:, 6]
        ], axis=1)

        all_states.append(states_14)
        all_actions.append(actions_4)

    X = np.concatenate(all_states)
    Y = np.concatenate(all_actions)
    return X, Y


class BCModel(nn.Module):
    """Same architecture as ActorCritic's shared trunk + actor_mean."""
    def __init__(self, obs_dim=14, act_dim=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./collected_episodes")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--save", default="./bc_policy.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading data from {args.data} ...")
    X, Y = load_all_episodes(args.data)
    print(f"  samples: {len(X)}, state_dim: {X.shape[1]}, action_dim: {Y.shape[1]}")

    # Normalize
    x_mean, x_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    y_mean, y_std = Y.mean(axis=0), Y.std(axis=0) + 1e-8
    X_norm = (X - x_mean) / x_std
    Y_norm = (Y - y_mean) / y_std

    dataset = TensorDataset(
        torch.tensor(X_norm, dtype=torch.float32),
        torch.tensor(Y_norm, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = BCModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

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

        scheduler.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}: loss={total_loss/len(loader):.4f}")

    # Save model + normalization stats
    torch.save({
        "model_state": model.state_dict(),
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
    }, args.save)
    print(f"Saved: {args.save}")


if __name__ == "__main__":
    main()
