"""Evaluate residual policy vs BC baseline with more episodes and randomized cubes."""

import argparse, warnings
import numpy as np
import torch
import mujoco
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


def run_episodes(env, pi05, pre, post, vis_dim, device, episodes,
                 residual_net=None, random_cube=False):
    """Run episodes. If residual_net is None, pure BC."""
    _feat = None
    def _hook(m, inp, out): nonlocal _feat; _feat = out.detach()

    successes = 0
    grasped = 0
    all_steps = []
    all_min_dist = []

    for ep in range(episodes):
        env.reset()
        if random_cube:
            _randomize_cube_anywhere(env)
        obs = env.get_obs_pi05()
        done = False
        min_dist = float('inf')

        while not done:
            ee = env.ee_position
            cube = env.cube_position
            min_dist = min(min_dist, float(np.linalg.norm(ee - cube)))

            batch = pre(build_batch(obs, device)); _feat = None
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = pi05.select_action(batch)
                bc = post(raw).squeeze(0).cpu().numpy()

            if residual_net is not None:
                vis_raw = _feat.squeeze(0) if _feat is not None else torch.zeros(vis_dim, device=device)
                if vis_raw.dim() == 2: vis_raw = vis_raw.mean(dim=0)
                vis_feat = vis_raw if vis_raw.dim() == 1 else vis_raw.squeeze(0)

                state7 = torch.tensor(np.concatenate([ee, cube, [env.gripper_width]]),
                                      dtype=torch.float32, device=device).unsqueeze(0)
                bc_xyz = torch.tensor(bc[:3], dtype=torch.float32, device=device).unsqueeze(0)
                vis_t = vis_feat.unsqueeze(0) if vis_feat.dim() == 1 else vis_feat

                with torch.no_grad():
                    delta = residual_net(vis_t, state7, bc_xyz).squeeze(0).cpu().numpy()
                bc_c = bc.copy()
                bc_c[:3] = bc[:3] + delta
                rl = np.array([bc_c[0], bc_c[1], bc_c[2], bc[6]])
            else:
                rl = np.array([bc[0], bc[1], bc[2], bc[6]])

            _, _, terminated, truncated, _ = env.step(rl)
            done = terminated or truncated
            obs = env.get_obs_pi05()

        all_steps.append(env._step_count)
        all_min_dist.append(min_dist)
        if env._success:
            successes += 1
        if env._grasped:
            grasped += 1

    return {
        "success": successes,
        "grasped": grasped,
        "avg_steps": np.mean(all_steps),
        "avg_min_dist": np.mean(all_min_dist),
        "median_min_dist": np.median(all_min_dist),
    }


def _randomize_cube_anywhere(env):
    """Place cube at random position within reachable area."""
    x = np.random.uniform(0.35, 0.55)
    y = np.random.uniform(-0.30, 0.30)
    z = env.CUBE_HALF  # on table
    cube_id = env._cube_body_id
    jnt_adr = env.model.jnt_qposadr[env.model.body_jntadr[cube_id]]
    env.data.qpos[jnt_adr:jnt_adr+3] = [x, y, z]
    # Also update init_cube and prev_dist for reward calculation
    env._init_cube = env.cube_position.copy()
    env._prev_dist = env._dist_to_target()
    mujoco.mj_forward(env.model, env.data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--residual-model", default=None,
                        help="path to trained residual_net.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--random-cube", action="store_true",
                        help="randomize cube position (test generalization)")
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Device: {device}")

    # ── Load PI0.5 ──
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

    _feat = None
    def _hook(m, inp, out): nonlocal _feat; _feat = out.detach()
    vis_dim = 512
    for target in ["action_in_proj", "state_proj", "multi_modal_projector.linear"]:
        for n, m in pi05.named_modules():
            if n.endswith(target):
                m.register_forward_hook(_hook)
                if hasattr(m, 'out_features'): vis_dim = m.out_features
                print(f"  Hooked {n}, vis_dim={vis_dim}")
                break
        if vis_dim > 100:
            break

    pre, post = make_pre_post_processors(policy_cfg=pi05.config, pretrained_path=args.checkpoint)
    print("PI0.5 loaded.")

    # ── Load residual net (optional) ──
    residual_net = None
    if args.residual_model:
        from train_residual import ResidualNet
        residual_net = ResidualNet(vis_dim).to(device)
        residual_net.load_state_dict(torch.load(args.residual_model, map_location=device))
        residual_net.eval()
        print(f"Residual model loaded from {args.residual_model}")

    env = PandaRLEnv()
    n = args.episodes
    cube_mode = "random" if args.random_cube else "fixed (3 positions)"

    # ── BC baseline ──
    print(f"\n{'='*50}")
    print(f"[BC Baseline] {n} episodes, cube: {cube_mode}")
    bc = run_episodes(env, pi05, pre, post, vis_dim, device, n,
                      residual_net=None, random_cube=args.random_cube)
    print(f"  Success:  {bc['success']}/{n} = {bc['success']/n*100:.1f}%")
    print(f"  Grasped:  {bc['grasped']}/{n} = {bc['grasped']/n*100:.1f}%")
    print(f"  Avg steps: {bc['avg_steps']:.0f}")
    print(f"  Min dist:  mean={bc['avg_min_dist']*100:.1f}cm  median={bc['median_min_dist']*100:.1f}cm")

    # ── Residual policy ──
    if residual_net is not None:
        print(f"\n{'='*50}")
        print(f"[BC + Residual] {n} episodes, cube: {cube_mode}")
        res = run_episodes(env, pi05, pre, post, vis_dim, device, n,
                           residual_net=residual_net, random_cube=args.random_cube)
        print(f"  Success:  {res['success']}/{n} = {res['success']/n*100:.1f}%")
        print(f"  Grasped:  {res['grasped']}/{n} = {res['grasped']/n*100:.1f}%")
        print(f"  Avg steps: {res['avg_steps']:.0f}")
        print(f"  Min dist:  mean={res['avg_min_dist']*100:.1f}cm  median={res['median_min_dist']*100:.1f}cm")

        delta_succ = res['success'] - bc['success']
        print(f"\n  Δ success: {delta_succ:+d} ({delta_succ/n*100:+.1f}%)")

    env.close()


if __name__ == "__main__":
    main()
