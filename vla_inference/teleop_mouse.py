"""Mouse-based Cartesian teleop + data recording for Franka Panda.

Control:
  Hold LEFT button + drag  → EE x/y (camera-frame: up=forward, right=right)
  Scroll wheel             → EE z up/down
  RIGHT button             → gripper toggle (press to close, release to open)

  A key    → save episode & next
  X key    → skip episode
  B key    → reset
  Y key    → exit

Records: state(15D) + action(8D) + front + wrist images.

Usage:
  python vla_inference/teleop_mouse.py --output collected_episodes/
"""

import argparse
import json
import math
import sys
import time
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from vla_inference.env.panda_joint_env import PandaJointEnv

# ── Shared state between mouse thread and main loop ──

class MouseState:
    def __init__(self):
        self.dx = 0.0      # accumulated mouse delta since last read
        self.dy = 0.0
        self.dz = 0.0      # scroll delta
        self.left = False  # left button held
        self.right = False
        self.grip_cmd = 0.0
        self.running = True

    def consume(self):
        """Read and reset accumulated deltas."""
        dx, dy, dz = self.dx, self.dy, self.dz
        self.dx = self.dy = self.dz = 0.0
        return dx, dy, dz


def start_mouse_listener(state: MouseState):
    """Background thread: capture global mouse events via pynput."""
    from pynput import mouse

    def on_move(x, y):
        pass  # handled via dx/dy in our own tracking

    def on_click(x, y, button, pressed):
        pass  # not using pynput buttons, using pygame's button state instead

    def on_scroll(x, y, dx, dy):
        state.dz += dy  # positive = scroll up

    # pynput listener — we use it only for scroll
    listener = mouse.Listener(on_scroll=on_scroll)
    listener.daemon = True
    listener.start()
    return listener


def _cam_frame_basis(azimuth_deg: float):
    """Camera forward/right vectors in world xy plane."""
    az = math.radians(azimuth_deg)
    fwd = np.array([-math.cos(az), -math.sin(az)])
    right = np.array([-math.sin(az), math.cos(az)])
    return fwd, right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./collected_episodes",
                        help="Output directory")
    parser.add_argument("--ctrl-hz", type=int, default=30,
                        help="Control/IK frequency")
    parser.add_argument("--record-hz", type=int, default=10,
                        help="Recording frequency")
    parser.add_argument("--cam-azimuth", type=float, default=180.0,
                        help="Camera azimuth (degrees)")
    parser.add_argument("--task", default="pick the red cube and place it on the pad")
    parser.add_argument("--ee-scale", type=float, default=15,
                        help="Mouse sensitivity: EE m per pixel (15mm/pixel for touchpad, try 0.003 for mouse)")
    args = parser.parse_args()

    env = PandaJointEnv(render_mode="human", scene="task")
    obs = env.reset()

    mouse_state = MouseState()
    scroll_listener = start_mouse_listener(mouse_state)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith("ep_")]
    ep_id = max([int(d.name.split("_")[1]) for d in existing], default=-1) + 1

    import pygame
    pygame.init()
    screen = pygame.display.set_mode((480, 320))
    pygame.display.set_caption("Mouse Teleop Panel")
    font = pygame.font.Font(None, 28)

    ctrl_interval = 1.0 / args.ctrl_hz
    record_every = args.ctrl_hz // args.record_hz
    fwd, right = _cam_frame_basis(args.cam_azimuth)

    MAX_FRAMES = 300
    data = {"states": [], "actions": [], "images_front": [], "images_wrist": []}
    active = False
    step_count = 0
    episode_count = 0
    clock = pygame.time.Clock()
    grip_toggled = False
    target_ee = None  # maintained independently to fight gravity

    def _new_episode():
        nonlocal active, data, step_count, grip_toggled, target_ee
        obs = env.reset()
        data = {"states": [], "actions": [], "images_front": [], "images_wrist": []}
        active = False
        step_count = 0
        grip_toggled = False
        target_ee = env.ee_position.copy()
        print(f"\n>>> Episode {ep_id} — drag mouse to start", flush=True)
        return obs

    obs = env.reset()
    target_ee = env.ee_position.copy()
    print("\n" + "=" * 50)
    print("Mouse: Left+drag=EE xy | Wheel=EE z | Right=grip | A:save X:skip B:reset Y:exit")
    print(f"Saving to: {out_dir}")
    print("=" * 50 + "\n")

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            # Read mouse state from pygame (faster than pynput for polling)
            mx, my = pygame.mouse.get_pos()
            left, mid, right_pressed = pygame.mouse.get_pressed()
            # Get relative motion since last frame
            dx, dy = pygame.mouse.get_rel()
            # Read keys
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
                grip_toggled = False
                mouse_state.grip_cmd = 0.0
                obs = _new_episode()
                continue

            # Auto-trigger on first mouse movement
            if not active and (abs(dx) + abs(dy) > 0):
                active = True
                print(" Recording!", flush=True)

            # EE velocity from mouse + scroll
            scroll_dz = mouse_state.consume()[2] * 0.003  # scroll sensitivity
            vx = fwd[0] * dy * (-1) * args.ee_scale  # mouse up → camera forward
            vy = fwd[1] * dy * (-1) * args.ee_scale
            vx += right[0] * dx * (-1) * args.ee_scale  # mouse right → camera right
            vy += right[1] * dx * (-1) * args.ee_scale

            # Gripper: toggle on right click
            if right_pressed and not grip_toggled:
                mouse_state.grip_cmd = 1.0 if mouse_state.grip_cmd < 0.5 else 0.0
                grip_toggled = True
            elif not right_pressed:
                grip_toggled = False

            grip = mouse_state.grip_cmd

            # Update target from mouse movement + scroll
            mouse_delta = np.array([vx, vy, scroll_dz]) * ctrl_interval
            # Clip max movement per frame to 1cm (prevent jumps)
            dist = np.linalg.norm(mouse_delta)
            if dist > 0.01:
                mouse_delta = mouse_delta / dist * 0.01
            # Deadzone: ignore sub-mm movements (relaxed for touchpad)
            if dist > 0.0001:
                target_ee = target_ee + mouse_delta
                target_ee[2] = max(0.01, target_ee[2])

            # Position correction: drive toward target, clamped to max 5cm step
            ee_err = target_ee - env.ee_position
            err_dist = np.linalg.norm(ee_err)
            if err_dist > 0.05:
                ee_err = ee_err / err_dist * 0.05
            if err_dist > 0.002:  # 2mm deadzone
                obs = env.step_ee(ee_err, grip)
            else:
                obs = env.step_ee(np.zeros(3), grip)
            action = env.get_action_from_ee(grip)

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

            # Render control panel
            screen.fill((30, 30, 30))
            lines = [
                f"Episode: {ep_id}  {'[REC]' if active else '[IDLE]'}",
                f"Frames: {len(data['states'])}/{MAX_FRAMES}",
                f"Gripper: {'CLOSED' if mouse_state.grip_cmd > 0.5 else 'OPEN'}",
                f"EE target: [{target_ee[0]:.3f}, {target_ee[1]:.3f}, {target_ee[2]:.3f}]",
                "",
                "L-drag: EE xy  Wheel: EE z  R: grip",
                "A:save  X:skip  B:reset  Y:exit",
            ]
            for i, line in enumerate(lines):
                surf = font.render(line, True, (200, 200, 200))
                screen.blit(surf, (20, 15 + i * 32))
            pygame.display.flip()

            clock.tick(args.ctrl_hz)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        pygame.quit()
        scroll_listener.stop()
        if len(data.get("states", [])) > 10:
            _save_episode(out_dir / f"ep_{ep_id:04d}", data, args.task)
            episode_count += 1

    print(f"\nDone. Saved {episode_count} episode(s) to {out_dir}")


def _save_episode(ep_dir: Path, data: dict, task: str = ""):
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
