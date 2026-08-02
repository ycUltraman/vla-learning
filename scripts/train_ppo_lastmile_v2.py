"""Online PPO for approach-phase positional refinement with PI0.5 vision features.

PPO activates when EE-cube < 20cm, outputs per-axis Δxyz correction.
Action = BC_xyz + Δxyz,  Δxy ∈ ±2cm,  Δz ∈ ±5mm.
Input: vis_proj(64D) + state(7D) + bc_xyz(3D) = 74D.
"""

import argparse, warnings, time
import numpy as np
import torch, torch.nn as nn
from torch.distributions import Normal
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


class LastMileActor(nn.Module):
    """74D → Δxyz (dx,dy,dz). xy ∈ ±2cm, z ∈ ±5mm. Per-axis output scales."""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 3))
        nn.init.normal_(self.net[-1].weight, std=0.001)
        nn.init.zeros_(self.net[-1].bias)
        self.scale = torch.tensor([0.02, 0.02, 0.005])  # xy=2cm, z=5mm

    def forward(self, x):
        raw = torch.tanh(self.net(x))
        return raw * self.scale.to(x.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--save", default="./lastmile_ppo.pt")
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Device: {device}")

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
    # Try multiple possible hook points, prefer the largest feature layer
    for target in ["action_in_proj", "state_proj", "multi_modal_projector.linear"]:
        for n, m in pi05.named_modules():
            if n.endswith(target):
                m.register_forward_hook(_hook)
                if hasattr(m, 'out_features'): vis_dim = m.out_features
                else: vis_dim = 512
                print(f"  Hooked {n}, vis_dim={vis_dim}")
                break
        if vis_dim > 100:
            break

    pre, post = make_pre_post_processors(policy_cfg=pi05.config, pretrained_path=args.checkpoint)
    print("PI0.5 loaded.")

    env = PandaRLEnv()
    vis_proj = nn.Linear(vis_dim, 64).to(device)  # compress vision features
    in_dim = 64 + 7 + 3  # projected_vision + state(ee,cube,grip) + bc_xyz
    actor = LastMileActor(in_dim=in_dim).to(device)
    log_std = nn.Parameter(torch.tensor([-4.5, -4.5, -6.0], device=device))  # xy: std≈0.011, z: std≈0.0025
    critic = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 1)).to(device)
    optim = torch.optim.Adam(list(actor.parameters())+list(vis_proj.parameters())+[log_std]+list(critic.parameters()), lr=args.lr)
    succ_cnt = 0

    for ep in range(args.episodes):
        env.reset(); obs = env.get_obs_pi05(); done = False; ep_reward = 0.0; t0 = time.time()
        buf_vis, buf_act, buf_rew, buf_lp, buf_val, buf_done = [], [], [], [], [], []

        while not done:
            ee, cube = env.ee_position, env.cube_position
            dist = float(np.linalg.norm(ee - cube))

            if dist < 0.20:
                # ── Approach phase: PI0.5 replans each step, PPO adds Δxyz correction ──
                for _ in range(20):
                    if done: break
                    batch = pre(build_batch(obs, device)); _feat = None
                    with torch.no_grad(), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        raw = pi05.select_action(batch)
                        bc = post(raw).squeeze(0).cpu().numpy()
                    vis_raw = _feat.squeeze(0) if _feat is not None else torch.zeros(1, vis_dim, device=device)
                    if vis_raw.dim() == 2: vis_raw = vis_raw.mean(dim=0)  # pool if (T,D)
                    vis_feat = vis_raw.squeeze(0) if vis_raw.dim() > 1 else vis_raw

                    ee, cube = env.ee_position, env.cube_position
                    prev_d = float(np.linalg.norm(ee - cube))
                    bc_xyz = torch.tensor(bc[:3], dtype=torch.float32, device=device)
                    state7 = torch.tensor(np.concatenate([ee, cube, [env.gripper_width]]),
                                          dtype=torch.float32, device=device)
                    vis_proj_feat = vis_proj(vis_feat)
                    inp = torch.cat([vis_proj_feat, state7, bc_xyz]).unsqueeze(0)

                    mean = actor(inp); std = torch.exp(log_std.clamp(-7, -1))
                    delta_raw = Normal(mean, std).rsample()
                    delta = delta_raw.clamp(-0.02, 0.02)  # safety clamp
                    lp = Normal(mean, std).log_prob(delta_raw).sum()
                    with torch.no_grad(): val = critic(inp).squeeze(-1)

                    d = delta.detach().cpu().numpy()
                    bc_c = bc.copy(); bc_c[:3] = bc[:3] + d
                    rl = np.array([bc_c[0], bc_c[1], bc_c[2], bc[6]])
                    _, reward, terminated, truncated, _ = env.step(rl)
                    done = terminated or truncated; ep_reward += reward
                    obs = env.get_obs_pi05()
                    new_d = float(np.linalg.norm(env.ee_position - env.cube_position))
                    r_total = 200.0 * (prev_d - new_d) + reward
                    if new_d > 0.20: break  # drifted out, BC takes over
                    if new_d < 0.015 or env._grasped: break  # close enough

                    buf_vis.append(inp.squeeze(0).detach()); buf_act.append(delta.detach())
                    buf_rew.append(r_total); buf_lp.append(lp.detach()); buf_val.append(val.detach()); buf_done.append(done)
                    if done: break
            else:
                batch = pre(build_batch(obs, device))
                with torch.no_grad(), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw = pi05.select_action(batch); bc = post(raw).squeeze(0).cpu().numpy()
                rl = np.array([bc[0], bc[1], bc[2], bc[6]])
                _, reward, terminated, truncated, _ = env.step(rl)
                done = terminated or truncated; ep_reward += reward; obs = env.get_obs_pi05()

        if env._success: succ_cnt += 1

        # PPO update
        n = len(buf_vis)
        if n > 2:
            vis_all = torch.stack(buf_vis); acts = torch.stack(buf_act)
            rews = torch.tensor(buf_rew, dtype=torch.float32, device=device)
            vals = torch.stack(buf_val).squeeze(-1)
            dones = torch.tensor(buf_done, dtype=torch.float32, device=device)
            lps = torch.stack(buf_lp)

            adv = torch.zeros(n, device=device); gae_val = 0.0
            for t in reversed(range(n)):
                nv = 0.0 if (buf_done[t] or t==n-1) else vals[t+1]
                gae_val = (rews[t]+0.99*nv-vals[t]) + 0.99*0.95*(1-buf_done[t])*gae_val
                adv[t] = gae_val
            rets = adv + vals; adv = (adv-adv.mean())/(adv.std()+1e-8)

            for _ in range(3):
                mean = actor(vis_all); std_val = torch.exp(log_std.clamp(-7, -1))
                dist = Normal(mean, std_val); new_lp = dist.log_prob(acts).sum(dim=-1)
                ratio = torch.exp(new_lp - lps)
                s1 = ratio*adv; s2 = torch.clamp(ratio,0.8,1.2)*adv
                pol_loss = -torch.min(s1, s2).mean()
                val_loss = nn.functional.mse_loss(critic(vis_all).squeeze(-1), rets)
                loss = pol_loss + 0.5*val_loss
                optim.zero_grad(); loss.backward(); optim.step()

        n_lm = len(buf_vis)
        dm = torch.stack(buf_act).abs().mean().item()*1000 if n_lm>0 else 0
        mr = np.mean(buf_rew) if n_lm>0 else 0
        print(f"Ep {ep:4d} | r: {ep_reward:+.1f} | lm={n_lm} | |D|={dm:.1f}mm | lm_r={mr:+.3f} | succ: {succ_cnt}/{ep+1}")

    print(f"\nFinal: {succ_cnt}/{args.episodes} = {succ_cnt/args.episodes*100:.0f}%")
    torch.save({"actor":actor.state_dict(),"log_std":log_std.data,"critic":critic.state_dict()}, args.save)
    env.close()

if __name__ == "__main__":
    main()
