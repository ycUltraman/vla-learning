"""SpaceMouse (3Dconnexion) 6-DOF Cartesian teleop for Franka Panda.

Control:
  SpaceMouse twist  → EE velocity (x/y/z, camera-frame)
  Left button       → gripper toggle
  Right button      → gripper toggle

  Keyboard:
    A → save episode & next
    X → skip
    B → reset
    Y → exit

Feels like holding the robot hand — no separate control panel needed.

Usage:
  python vla_inference/teleop_spacemouse.py --output collected_episodes/
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from vla_inference.env.panda_joint_env import PandaJointEnv


def _cam_frame_basis(azimuth_deg: float):
    az = math.radians(azimuth_deg)
    fwd = np.array([-math.cos(az), -math.sin(az)])
    right = np.array([-math.sin(az), math.cos(az)])
    return fwd, right


class SpaceMouseReader:
    """Non-blocking SpaceMouse state reader."""

    def __init__(self):
        self._last_state = None
        self._connected = False
        try:
            from pyspacemouse import open as sm_open, read as sm_read, list_devices
            devices = list_devices()
            if devices:
                sm_open()
                self._connected = True
                print(f"SpaceMouse connected: {devices[0]['name']}")
            else:
                print("No SpaceMouse found. Plug in and retry.")
        except Exception as e:
            print(f"SpaceMouse init error: {e}")

    @property
    def connected(self):
        return self._connected

    def read(self) -> dict | None:
        """Return dict with tx,ty,tz,rx,ry,rz,buttons or None."""
        if not self._connected:
            return None
        try:
            from pyspacemouse import read as sm_read
            state = sm_read()
            if state is not None:
                return {
                    "tx": state.x, "ty": state.y, "tz": state.z,
                    "rx": state.roll, "ry": state.pitch, "rz": state.yaw,
                    "buttons": state.buttons,
                }
        except Exception:
            pass
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./collected_episodes")
    parser.add_argument("--ctrl-hz", type=int, default=30)
    parser.add_argument("--record-hz", type=int, default=10)
    parser.add_argument("--cam-azimuth", type=float, default=180.0)
    parser.add_argument("--task", default="pick the red cube and place it on the pad")
    parser.add_argument("--vel-scale", type=float, default=0.03,
                        help="EE velocity scale (m/s per unit twist)")
    args = parser.parse_args()

    sm = SpaceMouseReader()
    if not sm.connected:
        sys.exit(1)

    env = PandaJointEnv(render_mode="human", scene="task")
    obs = env.reset()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith("ep_")]
    ep_id = max([int(d.name.split("_")[1]) for d in existing], default=-1) + 1

    import pygame
    pygame.init()
    pygame.display.set_mode((200, 100))
    pygame.display.set_caption("SpaceMouse | A:save X:skip B:reset Y:exit")

    ctrl_interval = 1.0 / args.ctrl_hz
    record_every = args.ctrl_hz // args.record_hz
    fwd, right = _cam_frame_basis(args.cam_azimuth)

    MAX_FRAMES = 300
    data = {"states": [], "actions": [], "images_front": [], "images_wrist": []}
    active = False
    step_count = 0
    episode_count = 0
    clock = pygame.time.Clock()
    target_ee = env.ee_position.copy()
    current_grip = 0.0
    grip_toggled = False

    def _new_episode():
        nonlocal active, data, step_count, target_ee, current_grip, grip_toggled
        obs = env.reset()
        data = {"states": [], "actions": [], "images_front": [], "images_wrist": []}
        active = False
        step_count = 0
        current_grip = 0.0
        grip_toggled = False
        target_ee = env.ee_position.copy()
        print(f"\n>>> Episode {ep_id} — twist SpaceMouse to start", flush=True)
        return obs

    print("\n" + "=" * 50)
    print("SpaceMouse: twist to move EE | L/R button: grip")
    print("Keyboard: A:save X:skip B:reset Y:exit")
    print(f"Saving to: {out_dir}")
    print("=" * 50 + "\n")

    try:
        while True:
            # Keyboard via pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            keys = pygame.key.get_pressed()
            if keys[pygame.K_y]:
                break
            if keys[pygame.K_x]:
                print(" SKIP")
                ep_id += 1
                obs = _new_episode()
                continue
            if keys[pygame.K_b]:
                print(" RESET")
                obs = _new_episode()
                continue

            # SpaceMouse state
            sm_state = sm.read()

            if sm_state is None:
                # No new data, maintain position against gravity
                ee_err = target_ee - env.ee_position
                err_dist = np.linalg.norm(ee_err)
                if err_dist > 0.05:
                    ee_err = ee_err / err_dist * 0.05
                if err_dist > 0.002:
                    obs = env.step_ee(ee_err, current_grip)
                else:
                    obs = env.step_ee(np.zeros(3), current_grip)
                clock.tick(args.ctrl_hz)
                continue

            # Auto-trigger
            if not active and (abs(sm_state["tx"]) + abs(sm_state["ty"]) + abs(sm_state["tz"]) > 0.01):
                active = True
                print(" Recording!", flush=True)

            # SpaceMouse twist → EE velocity (camera frame)
            sm_twist = np.array([sm_state["tx"], sm_state["ty"], sm_state["tz"]])
            if np.linalg.norm(sm_twist) > 1.0:
                sm_twist = sm_twist / np.linalg.norm(sm_twist)
            vx = fwd[0] * sm_twist[0] + right[0] * sm_twist[1]
            vy = fwd[1] * sm_twist[0] + right[1] * sm_twist[1]
            vz = sm_twist[2] * (-1)
            ee_vel = np.array([vx, vy, vz]) * args.vel_scale

            # Gripper: toggle on button press
            buttons = sm_state["buttons"]
            btn_pressed = (buttons[0] if len(buttons) > 0 else False) or \
                          (buttons[1] if len(buttons) > 1 else False)
            if btn_pressed and not grip_toggled:
                current_grip = 1.0 if current_grip < 0.5 else 0.0
                grip_toggled = True
            elif not btn_pressed:
                grip_toggled = False

            # Update target EE from velocity (camera frame → world)
            target_ee = target_ee + ee_vel * ctrl_interval
            target_ee[2] = max(0.01, target_ee[2])

            # Drive toward target (gravity compensation via position error)
            ee_err = target_ee - env.ee_position
            err_dist = np.linalg.norm(ee_err)
            if err_dist > 0.05:
                ee_err = ee_err / err_dist * 0.05
            if err_dist > 0.002:
                obs = env.step_ee(ee_err, current_grip)
            else:
                obs = env.step_ee(np.zeros(3), current_grip)

            action = env.get_action_from_ee(current_grip)
            step_count += 1

            if active and step_count % record_every == 0:
                data["states"].append(obs["observation.state"].copy())
                data["actions"].append(action.copy())
                data["images_front"].append(obs["observation.images.front"].copy())
                data["images_wrist"].append(obs["observation.images.wrist"].copy())

                n = len(data["states"])
                sys.stdout.write(f"\r  Frames: {n:4d}/{MAX_FRAMES}")
                sys.stdout.flush()

                if keys[pygame.K_a] and n > 10:
                    _save_episode(out_dir / f"ep_{ep_id:04d}", data, args.task)
                    episode_count += 1
                    ep_id += 1
                    obs = _new_episode()
                    continue
                if n >= MAX_FRAMES:
                    _save_episode(out_dir / f"ep_{ep_id:04d}", data, args.task)
                    episode_count += 1
                    ep_id += 1
                    obs = _new_episode()

            clock.tick(args.ctrl_hz)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        pygame.quit()
        if len(data.get("states", [])) > 10:
            _save_episode(out_dir / f"ep_{ep_id:04d}", data, args.task)
            episode_count += 1

    print(f"\nDone. Saved {episode_count} episode(s) to {out_dir}")


def _save_episode(ep_dir, data, task=""):
    ep_dir.mkdir(parents=True, exist_ok=True)
    n = len(data["actions"])
    np.savez_compressed(ep_dir / "trajectory.npz",
        states=np.array(data["states"], dtype=np.float32),
        actions=np.array(data["actions"], dtype=np.float32))
    np.savez_compressed(ep_dir / "images_front.npz",
        frames=np.array(data["images_front"], dtype=np.uint8))
    np.savez_compressed(ep_dir / "images_wrist.npz",
        frames=np.array(data["images_wrist"], dtype=np.uint8))
    with open(ep_dir / "meta.json", "w") as f:
        json.dump({"steps": n, "state_dim": 15, "action_dim": 8, "task": task}, f)
    print(f"\n  Saved {n} frames → {ep_dir}")


if __name__ == "__main__":
    main()
