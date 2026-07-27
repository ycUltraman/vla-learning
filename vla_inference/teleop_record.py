"""Gamepad teleoperation + data recording for Franka Panda.

Cartesian (end-effector) control via Jacobian IK:
  Left stick X/Y  → EE move in horizontal plane
  Right stick Y   → EE move up/down
  LB / RB         → gripper open / close
  RT (trigger)    → EE fine down
  LT (trigger)    → EE fine up
  X button        → skip / fail current episode
  Y button        → save & exit

Flow: auto reset → 2s wait → control arm → complete task → auto save → repeat

Records joint action format:
  state:  15D [joint1..7, gripper_width, ee_xyz, ee_quat_wxyz]
  action:  8D [joint1..7 targets, gripper_cmd 0-1]
  images: front (640x480) + wrist (640x480)

Usage:
  python vla_inference/teleop_record.py --output collected_episodes/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from vla_inference.env.panda_joint_env import PandaJointEnv


def init_gamepad():
    """Initialize pygame joystick. Returns joystick object or None."""
    try:
        import pygame
    except ImportError:
        print("pygame not installed. Run: pip install pygame")
        return None

    pygame.init()
    pygame.joystick.init()
    n = pygame.joystick.get_count()
    if n == 0:
        print("No gamepad detected. Connect a controller and retry.")
        pygame.quit()
        return None

    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"Gamepad connected: {joy.get_name()}")
    return joy


def read_gamepad(joy) -> dict:
    """Read gamepad state into a dict of axis/button values."""
    import pygame

    pygame.event.pump()
    return {
        # Left stick
        "lx": joy.get_axis(0),   # ±1
        "ly": joy.get_axis(1),   # ±1
        # Right stick
        "rx": joy.get_axis(2),   # ±1 (or 3 on some controllers)
        "ry": joy.get_axis(3),   # ±1 (or 4 on some controllers)
        # Triggers (may be axes or buttons depending on OS)
        "lt": joy.get_axis(4) if joy.get_numaxes() > 4 else 0.0,  # -1..1
        "rt": joy.get_axis(5) if joy.get_numaxes() > 5 else 0.0,  # -1..1
        # Buttons
        "a": joy.get_button(0),
        "b": joy.get_button(1),
        "x": joy.get_button(2),
        "y": joy.get_button(3),
        "lb": joy.get_button(4),
        "rb": joy.get_button(5),
        "back": joy.get_button(6),
        "start": joy.get_button(7),
        # D-pad (hat)
        "dpad_x": 0,
        "dpad_y": 0,
    }


def _deadzone_quadratic(val: float, deadzone: float = 0.15) -> float:
    """Apply deadzone + quadratic curve for fine control at small deflections."""
    if abs(val) < deadzone:
        return 0.0
    # Rescale [deadzone, 1.0] → [0, 1.0], then square for fine control
    sign = 1.0 if val > 0 else -1.0
    scaled = (abs(val) - deadzone) / (1.0 - deadzone)
    return sign * scaled * scaled


def gamepad_to_ee_cmd(gp: dict, current_grip: float, cam_azimuth: float = 180.0) -> tuple:
    """Convert gamepad to EE velocity in CAMERA frame.
    Stick forward = move EE deeper into scene (away from camera).
    Stick right = move EE to screen right.

    Returns: (ee_vel_xyz: np(3,), gripper_cmd: float)
    """
    import math
    VEL_SCALE = 0.5  # 10mm/s at full stick

    # Camera frame basis in world (horizontal plane)
    az = math.radians(cam_azimuth)
    # Camera forward in world: direction from camera toward lookat (horizontal)
    cam_fwd = np.array([-math.cos(az), -math.sin(az)])
    cam_right = np.array([-math.sin(az), math.cos(az)])

    # Joystick: ly forward/back, lx left/right → world x,y
    sx = _deadzone_quadratic(gp["ly"]) * (-1)   # stick forward = camera forward
    sy = _deadzone_quadratic(gp["lx"]) * (-1)   # stick right = camera right
    v_xy = (cam_fwd * sx + cam_right * sy) * VEL_SCALE

    vx, vy = v_xy[0], v_xy[1]
    vz = _deadzone_quadratic(gp["ry"]) * (-1) * VEL_SCALE

    # Gripper: binary — RB=closed, LB=open, neither=hold
    if gp["rb"]:
        grip_cmd = 1.0
    elif gp["lb"]:
        grip_cmd = 0.0
    else:
        grip_cmd = current_grip

    return np.array([vx, vy, vz]), grip_cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./collected_episodes",
                        help="Output directory for recorded episodes")
    parser.add_argument("--fps", type=int, default=10,
                        help="Recording frequency (Hz)")
    parser.add_argument("--ctrl-hz", type=int, default=30,
                        help="Control/IK frequency (Hz), higher = smoother teleop")
    parser.add_argument("--task", default="move to the red cube",
                        help="Task description for training")
    parser.add_argument("--cam-azimuth", type=float, default=180.0,
                        help="Camera azimuth (degrees), must match env camera")
    args = parser.parse_args()

    joy = init_gamepad()
    if joy is None:
        sys.exit(1)

    env = PandaJointEnv(render_mode="human", scene="task")
    obs = env.reset()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find next episode id
    existing = [d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith("ep_")]
    ep_id = max([int(d.name.split("_")[1]) for d in existing], default=-1) + 1

    MAX_STEPS = 300   # ~300 recorded frames per episode
    IDLE_THRESH = 0.1
    ctrl_interval = 1.0 / args.ctrl_hz      # IK/control period
    record_every = args.ctrl_hz // args.fps  # record every N control steps
    current_grip = 0.0
    episode_count = 0
    last_ctrl = time.perf_counter()
    data = {"states": [], "actions": [], "images_front": [], "images_wrist": []}
    step_in_ep = 0
    active = False

    print("\n" + "=" * 50)
    print("Move stick to start | A: save & next | B: reset | X: skip | Y: exit")
    print(f"Saving to: {out_dir}")
    print("=" * 50 + "\n")

    import pygame
    clock = pygame.time.Clock()

    def _new_episode():
        nonlocal step_in_ep, active, data, current_grip
        current_grip = 0.0
        obs = env.reset()
        data = {"states": [], "actions": [], "images_front": [], "images_wrist": []}
        step_in_ep = 0
        active = False
        print(f"\n>>> Episode {ep_id} — move stick to start", flush=True)
        return obs, data

    obs, data = _new_episode()

    try:
        while True:
            gp = read_gamepad(joy)
            t_now = time.perf_counter()

            if gp["x"]:
                print(" SKIP")
                ep_id += 1
                current_grip = 0.0
                obs, data = _new_episode()
                continue

            if gp["b"]:
                print(" RESET")
                obs, data = _new_episode()
                continue

            if gp["y"]:
                break

            # Auto-trigger recording on first significant input
            if not active:
                if (abs(gp["lx"]) > IDLE_THRESH or abs(gp["ly"]) > IDLE_THRESH
                        or abs(gp["ry"]) > IDLE_THRESH or gp["lb"] or gp["rb"]):
                    active = True
                    print(" Recording!", flush=True)

            ee_vel, grip_cmd = gamepad_to_ee_cmd(gp, current_grip, args.cam_azimuth)
            current_grip = grip_cmd
            ee_delta = ee_vel * ctrl_interval

            if t_now - last_ctrl >= ctrl_interval:
                if not active:
                    env.compute_target_joints(ee_delta)
                    obs = env.step(np.concatenate([env._ee_target_joints, [grip_cmd]]))
                    last_ctrl = t_now
                    continue

                env.compute_target_joints(ee_delta)
                action = np.concatenate([env._ee_target_joints.copy(), [grip_cmd]])
                obs = env.step(action)
                last_ctrl = t_now
                step_in_ep += 1

                if step_in_ep % record_every == 0:
                    data["states"].append(obs["observation.state"].copy())
                    data["images_front"].append(obs["observation.images.front"].copy())
                    data["images_wrist"].append(obs["observation.images.wrist"].copy())
                    data["actions"].append(action.copy())

                n = len(data["states"])
                sys.stdout.write(f"\r  Frames: {n:4d}/{MAX_STEPS}")
                sys.stdout.flush()

                if gp["a"] and n > 10:
                    _save_episode(out_dir / f"ep_{ep_id:04d}", data, args.task)
                    episode_count += 1
                    ep_id += 1
                    obs, data = _new_episode()
                    continue

                if n >= MAX_STEPS:
                    _save_episode(out_dir / f"ep_{ep_id:04d}", data, args.task)
                    episode_count += 1
                    ep_id += 1
                    obs, data = _new_episode()

            clock.tick(60)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        try: pygame.quit()
        except Exception: pass

        if len(data["states"]) > 10:
            _save_episode(out_dir / f"ep_{ep_id:04d}", data, args.task)
            episode_count += 1

    print(f"\nDone. Saved {episode_count} episode(s) to {out_dir}")


def _save_episode(ep_dir: Path, data: dict, task: str = "", fps: int = 10):
    """Save one episode as npz (Windows-compatible). Convert to LeRobot on server."""
    ep_dir.mkdir(parents=True, exist_ok=True)
    n = len(data["actions"])
    states = np.array(data["states"], dtype=np.float32)
    actions = np.array(data["actions"], dtype=np.float32)
    front = np.array(data["images_front"], dtype=np.uint8)
    wrist = np.array(data["images_wrist"], dtype=np.uint8)

    np.savez_compressed(ep_dir / "trajectory.npz", states=states, actions=actions)
    np.savez_compressed(ep_dir / "images_front.npz", frames=front)
    np.savez_compressed(ep_dir / "images_wrist.npz", frames=wrist)

    with open(ep_dir / "meta.json", "w") as f:
        json.dump({"steps": n, "state_dim": 15, "action_dim": 8, "task": task}, f)

    print(f"\n  Saved {n} frames → {ep_dir}")


if __name__ == "__main__":
    main()
