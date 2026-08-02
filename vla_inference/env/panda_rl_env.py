"""Gym-compatible Franka Panda RL environment for pick-and-place.

State (14D): joint_pos[7] + ee_pos[3] + cube_pos[3] + gripper_width[1]
Action (4D):  [dx, dy, dz, gripper_cmd]  gripper_cmd: 0=open, 1=closed

Reward: dense distance-based + grasp bonus + success bonus.
"""

import numpy as np
import mujoco
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_TASK_SCENE = str(
    _PROJECT_ROOT / "mujoco_menagerie" / "franka_emika_panda" / "scene_task.xml"
)
# Fallback: check common locations for the scene file
if not Path(_TASK_SCENE).exists():
    _ALT = Path("/root/autodl-tmp/mujoco_menagerie/franka_emika_panda/scene_task.xml")
    if _ALT.exists():
        _TASK_SCENE = str(_ALT)


class PandaRLEnv:
    """Gym-style environment for PPO training on the red cube pick task."""

    HOME_QPOS = np.array(
        [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04]
    )
    EE_OFFSET = np.array([0.0, 0.0, 0.15])
    CUBE_HALF = 0.03

    # Task config
    MAX_STEPS = 300
    SUCCESS_Z = 0.10    # cube z > 0.10 = success (cube starts at 0.03, lifted >7cm)

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(_TASK_SCENE)
        self.data = mujoco.MjData(self.model)
        self._resolve_ids()

        # Image rendering for PI0.5 input
        self._renderer = mujoco.Renderer(self.model, 480, 640)
        self._cam_front = mujoco.MjvCamera()
        self._cam_front.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._cam_front.lookat = np.array([0.6, -0.1, 0.32])
        self._cam_front.distance = 0.50
        self._cam_front.azimuth = 180
        self._cam_front.elevation = -25.0
        self._wrist_cam_name = "wrist_cam"

        self._ee_target_joints = None
        self._step_count = 0

    # ── Gym API ────────────────────────────────────────────────

    def reset(self, seed=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:9] = self.HOME_QPOS
        self._randomize_cube()
        mujoco.mj_forward(self.model, self.data)
        self._ee_target_joints = self.joint_positions.copy()
        self._step_count = 0
        self._grasped = False
        self._success = False
        self._init_cube = self.cube_position.copy()  # fixed target, never changes
        self._prev_dist = self._dist_to_target()
        self._prev_ee = self.ee_position.copy()
        self._prev_grip = self.gripper_width
        # Reward breakdown tracking
        self._rew_progress = 0.0
        self._rew_pose = 0.0
        self._rew_grasp = 0.0
        self._rew_success = 0.0
        self._rew_attempt = 0.0
        self._rew_step = 0.0
        self._rew_terminal = 0.0
        self._min_pose_dist = float('inf')
        self._ever_near_grasp = False
        self._idle_near = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """action: [dx, dy, dz, gripper_cmd]"""
        action = np.asarray(action, dtype=np.float64)
        ee_delta = action[:3]
        gripper_cmd = float(np.clip(action[3], 0.0, 1.0))

        # Track gripper transition for grasp-attempt detection
        prev_grip = self.gripper_width

        # Apply EE delta via 6D IK — 3 passes like PandaJointEnv.apply_ee_delta
        for _ in range(3):
            self._compute_ik(ee_delta / 3.0)
            self.data.ctrl[:7] = self._ee_target_joints
            self.data.ctrl[7] = np.clip((1.0 - gripper_cmd) * 255.0, 0.0, 255.0)
            mujoco.mj_step(self.model, self.data, nstep=50)
        mujoco.mj_forward(self.model, self.data)

        self._step_count += 1
        self._check_grasp()
        obs = self._get_obs()

        # Detect grasp attempt: gripper transitioned from open to closed
        grasp_attempt = (prev_grip > 0.05 and self.gripper_width < 0.04
                         and not self._grasped and not self._success)
        if grasp_attempt:
            self._grasp_attempted = True

        reward = self._compute_reward(grasp_attempt)
        terminated = self._success
        truncated = self._step_count >= self.MAX_STEPS

        # Terminal penalty: timeout without success
        if truncated and not self._success:
            reward -= 5.0   # light — success +500 dominates
            self._rew_terminal -= 5.0

        # Early termination: in grasp zone but no progress for 100 steps
        _xy = float(np.linalg.norm(self.ee_position[:2] - self._init_cube[:2]))
        _z = abs(self.ee_position[2] - (self._init_cube[2] + self.CUBE_HALF + 0.01))
        _pose = float(np.sqrt(_xy**2 + _z**2))
        if _pose < 0.12 and not self._grasped and not self._success:
            self._idle_near += 1
        else:
            self._idle_near = 0
        if self._idle_near > 100:
            truncated = True

        return obs, reward, terminated, truncated, {}

    # ── State accessors ─────────────────────────────────────────

    @property
    def joint_positions(self) -> np.ndarray:
        return np.array([self.data.qpos[jid] for jid in self._arm_joint_qpos_ids])

    @property
    def gripper_width(self) -> float:
        return float(
            self.data.qpos[self._finger1_qpos_id]
            + self.data.qpos[self._finger2_qpos_id]
        )

    @property
    def ee_position(self) -> np.ndarray:
        xmat = self.data.xmat[self._hand_body_id].reshape(3, 3)
        return self.data.xpos[self._hand_body_id] + xmat @ self.EE_OFFSET

    @property
    def cube_position(self) -> np.ndarray:
        return self.data.xpos[self._cube_body_id].copy()

    # ── Internal ────────────────────────────────────────────────

    def _resolve_ids(self):
        m = self.model
        n2i = lambda t, n: mujoco.mj_name2id(m, t, n)
        self._arm_joint_qpos_ids = [
            m.jnt_qposadr[n2i(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]
            for i in range(1, 8)
        ]
        self._finger1_qpos_id = m.jnt_qposadr[
            n2i(mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        ]
        self._finger2_qpos_id = m.jnt_qposadr[
            n2i(mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
        ]
        self._hand_body_id = n2i(mujoco.mjtObj.mjOBJ_BODY, "hand")
        self._cube_body_id = n2i(mujoco.mjtObj.mjOBJ_BODY, "red_cube")

    # Training data cube positions (must match BC training distribution)
    CUBE_POSITIONS = [
        np.array([0.45, -0.20, 0.03]),
        np.array([0.45,  0.00, 0.03]),
        np.array([0.45,  0.20, 0.03]),
    ]

    def _randomize_cube(self):
        """Randomly pick one of the 3 BC training positions."""
        pos = self.CUBE_POSITIONS[np.random.randint(0, 3)]
        self.data.qpos[
            self.model.jnt_qposadr[
                self.model.body_jntadr[self._cube_body_id]
            ]:self.model.jnt_qposadr[
                self.model.body_jntadr[self._cube_body_id]
            ] + 3
        ] = [pos[0], pos[1], self.CUBE_HALF]

    def _get_obs(self) -> np.ndarray:
        """14D state vector (for state-only RL)."""
        return np.concatenate([
            self.joint_positions,      # 7
            self.ee_position,           # 3
            self.cube_position,         # 3
            [self.gripper_width],       # 1
        ]).astype(np.float32)

    def get_obs_pi05(self) -> dict:
        """Return PI0.5-compatible observation dict (15D state + images)."""
        # 15D state = [j1..j7, grip_w, ee_xyz, quat_wxyz] (matches training)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.xmat[self._hand_body_id])
        state = np.concatenate([
            self.joint_positions,       # 7
            [self.gripper_width],       # 1
            self.ee_position,           # 3
            quat,                       # 4
        ]).astype(np.float32)

        # Render images
        self._renderer.update_scene(self.data, camera=self._cam_front)
        front = self._renderer.render()
        self._renderer.update_scene(self.data, camera=self._wrist_cam_name)
        wrist = self._renderer.render()

        return {
            "observation.state": state,
            "observation.images.front": front,
            "observation.images.wrist": wrist,
        }

    def _compute_ik(self, ee_delta: np.ndarray):
        """6D DLS IK — same as panda_joint_env."""
        ee_delta = np.asarray(ee_delta, dtype=np.float64)
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, self._hand_body_id)
        Jp, Jr = jac_pos[:, :7], jac_rot[:, :7]
        err = np.concatenate([ee_delta, np.zeros(3)])
        J = np.vstack([Jp, Jr])
        dq = J.T @ np.linalg.inv(J @ J.T + 0.03 * 0.03 * np.eye(6)) @ err
        self._ee_target_joints[:7] += dq
        for i in range(7):
            lo, hi = self.model.jnt_range[i]
            self._ee_target_joints[i] = np.clip(self._ee_target_joints[i], lo, hi)

    def _check_grasp(self):
        """Cube Z height — grasped resets if cube drops back to table."""
        cz = self.cube_position[2]
        self._grasped = cz > 0.06      # reset if cube falls back
        if cz > 0.10:
            self._success = True        # permanent once achieved

    def _dist_to_target(self) -> float:
        """3D distance from EE to INITIAL cube position (fixed target)."""
        target = np.array([
            self._init_cube[0],
            self._init_cube[1],
            self._init_cube[2] + self.CUBE_HALF + 0.01,
        ])
        return float(np.linalg.norm(self.ee_position - target))

    def _compute_reward(self, grasp_attempt: bool = False) -> float:
        """Reward: progress outside zone or simple distance-based."""
        # Simple mode for state-only PPO: pure distance + success
        if getattr(self, '_simple_reward', False):
            dist = self._dist_to_target()
            progress = self._prev_dist - dist
            r = 2.0 * progress  # strong direction signal
            r -= 0.3 * dist      # distance penalty
            if self._grasped and not getattr(self, '_grasp_bonus_given', False):
                r += 50.0
                self._grasp_bonus_given = True
            if self._success and not getattr(self, '_success_bonus_given', False):
                r += 500.0
                self._success_bonus_given = True
            r -= 0.005
            self._prev_dist = dist
            return float(r)

        # Complex mode for PI0.5 residual PPO
        dist = self._dist_to_target()
        xy_dist = float(np.linalg.norm(self.ee_position[:2] - self._init_cube[:2]))
        z_target = self._init_cube[2] + self.CUBE_HALF + 0.01
        z_err = abs(self.ee_position[2] - z_target)
        _pose = float(np.sqrt(xy_dist**2 + z_err**2))

        # Outside grasp zone: weak progress reward
        if _pose > 0.12:
            progress = np.clip(self._prev_dist - dist, -0.01, 0.01)
            r = 0.2 * progress
            prev_z_err = abs(self._prev_ee[2] - z_target)
            r += 3.0 * np.clip(prev_z_err - z_err, -0.005, 0.005)
        else:
            r = 0.0  # freeze — must grasp to earn
        self._rew_progress += r

        # Track min distance for debugging
        self._min_pose_dist = min(self._min_pose_dist, _pose)
        if _pose < 0.12:
            self._ever_near_grasp = True

        # Grip reward inside zone: closing = good
        grip_change = self._prev_grip - self.gripper_width
        if _pose < 0.12:
            r += 50.0 * grip_change
            self._rew_pose += 50.0 * grip_change

        # Grasp bonus (one-time)
        if self._grasped and not getattr(self, '_grasp_bonus_given', False):
            r += 100.0
            self._rew_grasp += 100.0
            self._grasp_bonus_given = True

        # Flat reward for success — simple and correct
        if self._success and not getattr(self, '_success_bonus_given', False):
            r += 1000.0
            self._rew_success += 1000.0
            self._success_bonus_given = True

        # Grasp attempt penalty (one-time, light)
        if grasp_attempt:
            penalty = 0.2 if dist < 0.10 else 1.0
            r -= penalty
            self._rew_attempt -= penalty

        r -= 0.005  # step cost
        self._prev_dist = dist
        self._prev_grip = self.gripper_width
        return float(r)

    def close(self):
        pass
