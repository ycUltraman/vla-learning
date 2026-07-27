"""GUI-based Cartesian teleop + recording for Franka Panda.

A clean control panel: drag in the pad to move EE x/y,
scroll wheel for z, buttons for gripper/save/skip/reset.

Usage:
  python vla_inference/teleop_gui.py --output collected_episodes/
"""

import argparse
import json
import math
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from vla_inference.env.panda_joint_env import PandaJointEnv


@dataclass
class MouseState:
    dx: float = 0.0
    dy: float = 0.0
    scroll: float = 0.0
    grip_cmd: float = 0.0
    active: bool = False
    running: bool = True
    keys: set = field(default_factory=set)


def _cam_basis(azimuth_deg):
    az = math.radians(azimuth_deg)
    fwd = np.array([-math.cos(az), -math.sin(az)])
    right = np.array([-math.sin(az), math.cos(az)])
    return fwd, right


def env_loop(env, state, args, out_dir):
    """Runs in bg thread: MuJoCo env step + recording."""
    fwd, right = _cam_basis(args.cam_azimuth)
    ctrl_interval = 1.0 / args.ctrl_hz
    record_every = args.ctrl_hz // args.record_hz
    target_ee = env.ee_position.copy()
    current_grip = 0.0

    data = {"states": [], "actions": [], "images_front": [], "images_wrist": []}
    active = False
    step_count = 0
    ep_id = 0
    episode_count = 0
    grip_toggled = False
    last_right = False

    def _new_ep():
        nonlocal active, data, step_count, target_ee, current_grip, grip_toggled
        obs = env.reset()
        data = {"states": [], "actions": [], "images_front": [], "images_wrist": []}
        active = False
        step_count = 0
        current_grip = 0.0
        grip_toggled = False
        target_ee = env.ee_position.copy()
        return obs

    obs = _new_ep()

    while state.running:
        t0 = time.perf_counter()

        k = state.keys
        if "y" in k:
            state.running = False
            break
        if "x" in k:
            state.keys.discard("x")
            ep_id += 1
            obs = _new_ep()
            continue
        if "b" in k:
            state.keys.discard("b")
            obs = _new_ep()
            continue

        # Gripper toggle on right-click
        right_now = (k and "grip" in k)
        if right_now and not grip_toggled:
            state.grip_cmd = 0.0 if state.grip_cmd > 0.5 else 1.0
            grip_toggled = True
        elif not right_now:
            grip_toggled = False
        state.keys.discard("grip")

        # Auto-trigger on first mouse drag
        if not active and (abs(state.dx) > 2 or abs(state.dy) > 2):
            active = True
            state.active = True  # signal GUI

        # EE velocity from mouse (camera frame)
        ee_vel = np.zeros(3)
        ee_vel[0] = fwd[0] * state.dy + right[0] * state.dx
        ee_vel[1] = fwd[1] * state.dy + right[1] * state.dx
        ee_vel[2] = state.scroll
        ee_vel *= args.vel_scale / 100.0  # pixel → m/s
        state.dx = state.dy = state.scroll = 0.0

        # Update target
        target_ee = target_ee + ee_vel * ctrl_interval
        target_ee[2] = max(0.01, target_ee[2])

        # Position correction (gravity compensation)
        ee_err = target_ee - env.ee_position
        err_dist = np.linalg.norm(ee_err)
        if err_dist > 0.05:
            ee_err = ee_err / err_dist * 0.05
        if err_dist > 0.002:
            obs = env.step_ee(ee_err, state.grip_cmd)
        else:
            obs = env.step_ee(np.zeros(3), state.grip_cmd)

        action = env.get_action_from_ee(state.grip_cmd)
        step_count += 1

        if active and step_count % record_every == 0:
            data["states"].append(obs["observation.state"].copy())
            data["actions"].append(action.copy())
            data["images_front"].append(obs["observation.images.front"].copy())
            data["images_wrist"].append(obs["observation.images.wrist"].copy())
            n = len(data["states"])

            if "a" in k and n > 10:
                state.keys.discard("a")
                _save_ep(out_dir / f"ep_{ep_id:04d}", data, args.task)
                episode_count += 1
                ep_id += 1
                obs = _new_ep()

            if n >= args.max_frames:
                _save_ep(out_dir / f"ep_{ep_id:04d}", data, args.task)
                episode_count += 1
                ep_id += 1
                obs = _new_ep()

        elapsed = time.perf_counter() - t0
        sleep_t = ctrl_interval - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    env.close()
    if len(data.get("states", [])) > 10:
        _save_ep(out_dir / f"ep_{ep_id:04d}", data, args.task)
        episode_count += 1
    print(f"\nSaved {episode_count} episodes")


def _save_ep(ep_dir, data, task=""):
    ep_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ep_dir / "trajectory.npz",
        states=np.array(data["states"], dtype=np.float32),
        actions=np.array(data["actions"], dtype=np.float32))
    np.savez_compressed(ep_dir / "images_front.npz",
        frames=np.array(data["images_front"], dtype=np.uint8))
    np.savez_compressed(ep_dir / "images_wrist.npz",
        frames=np.array(data["images_wrist"], dtype=np.uint8))
    with open(ep_dir / "meta.json", "w") as f:
        json.dump({"steps": len(data["states"]), "state_dim": 15, "action_dim": 8, "task": task}, f)
    print(f"  Saved ep → {ep_dir}")


# ── GUI ────────────────────────────────────────────────────────

def build_gui(state: MouseState):
    root = tk.Tk()
    root.title("Franka Teleop")
    root.geometry("340x440")
    root.configure(bg="#2b2b2b")

    fg, bg, accent = "#e0e0e0", "#2b2b2b", "#3a3a3a"

    # Main control pad
    pad = tk.Canvas(root, width=280, height=260, bg="#1a1a1a",
                    highlightthickness=2, highlightbackground="#555")
    pad.pack(pady=(15, 5))

    # Crosshair
    pad.create_line(140, 20, 140, 240, fill="#444", dash=(4, 8))
    pad.create_line(20, 130, 260, 130, fill="#444", dash=(4, 8))
    pad.create_text(140, 8, text="forward ↑", fill="#666", font=("", 8))
    pad.create_text(140, 252, text="back ↓", fill="#666", font=("", 8))
    pad.create_text(8, 130, text="← left", fill="#666", font=("", 8))
    pad.create_text(272, 130, text="right →", fill="#666", font=("", 8))
    cursor = pad.create_oval(134, 124, 146, 136, fill="#5af", outline="")

    def on_mouse_drag(event):
        pad.move("all", event.x - 140, event.y - 130)
        return "break"

    def on_mouse_move(event):
        """Track dx,dy in the pad while left button held."""
        if not (event.state & 0x0100):  # left button
            return
        dx = event.x - 140
        dy = 130 - event.y
        state.dx += dx
        state.dy += dy
        # Move cursor
        cx = max(10, min(270, 140 + dx))
        cy = max(10, min(250, 130 - dy))
        pad.coords(cursor, cx-6, cy-6, cx+6, cy+6)

    pad.bind("<Motion>", on_mouse_move, add="+")
    pad.bind("<B1-Motion>", on_mouse_move)

    # Scroll wheel → Z
    def on_scroll(event):
        state.scroll += -event.delta / 120.0
    pad.bind("<MouseWheel>", on_scroll)
    # Right click → gripper toggle
    def on_right(event):
        state.keys.add("grip")
    pad.bind("<Button-3>", on_right)

    # Label
    status_label = tk.Label(root, text="Drag in pad to move EE  |  Scroll = Z  |  Right-click = grip",
                           bg=bg, fg=fg, font=("", 9))
    status_label.pack(pady=(5, 8))

    # Buttons
    btn_frame = tk.Frame(root, bg=bg)
    btn_frame.pack()

    def make_btn(text, key, color, width=8):
        btn = tk.Button(btn_frame, text=text, width=width, height=2,
                        bg=color, fg="#fff", font=("", 10, "bold"),
                        command=lambda: state.keys.add(key))
        btn.pack(side=tk.LEFT, padx=4)
        return btn

    make_btn("OPEN", "open", "#4a9")   # gripper open
    make_btn("CLOSE", "close", "#c44")  # gripper close
    make_btn("SKIP", "x", "#666")       # skip
    make_btn("RESET", "b", "#666")      # reset
    make_btn("SAVE", "a", "#5a5", width=5)

    # Bind open/close to state
    def poll_keys():
        if "open" in state.keys:
            state.keys.discard("open")
            state.grip_cmd = 0.0
        if "close" in state.keys:
            state.keys.discard("close")
            state.grip_cmd = 1.0
        if state.active:
            status_label.config(text="● RECORDING  |  Drag to move  |  Scroll = Z")
        root.after(50, poll_keys)

    poll_keys()
    root.protocol("WM_DELETE_WINDOW", lambda: state.keys.add("y"))
    root.mainloop()
    state.running = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./collected_episodes")
    parser.add_argument("--ctrl-hz", type=int, default=30)
    parser.add_argument("--record-hz", type=int, default=10)
    parser.add_argument("--cam-azimuth", type=float, default=180.0)
    parser.add_argument("--vel-scale", type=float, default=1.0,
                        help="Mouse sensitivity")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--task", default="pick the red cube and place it on the pad")
    args = parser.parse_args()

    state = MouseState()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = PandaJointEnv(render_mode="human", scene="task")
    env.reset()

    # Start MuJoCo env loop in background thread
    env_thread = threading.Thread(
        target=env_loop, args=(env, state, args, out_dir), daemon=True
    )
    env_thread.start()

    # GUI runs in main thread
    build_gui(state)

    env_thread.join(timeout=2)


if __name__ == "__main__":
    main()
