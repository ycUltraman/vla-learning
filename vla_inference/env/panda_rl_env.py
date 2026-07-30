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
    MAX_STEPS = 200
    GRASP_Z = 0.06      # z height to consider "reached cube"
    LIFT_Z = 0.15       # z height to consider "lifted cube"
    GRASP_THRESH = 0.03  # xy distance to consider "above cube"
    SUCCESS_Z = 0.12     # cube z above this = success

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(_TASK_SCENE)
        self.data = mujoco.MjData(self.model)
        self._resolve_ids()
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
        self._cube_attached = False
        self._grasped = False
        self._success = False
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """action: [dx, dy, dz, gripper_cmd]"""
        action = np.asarray(action, dtype=np.float64)
        ee_delta = action[:3]
        gripper_cmd = float(np.clip(action[3], 0.0, 1.0))

        # Apply EE delta via 6D IK
        self._compute_ik(ee_delta)
        self.data.ctrl[:7] = self._ee_target_joints
        self.data.ctrl[7] = np.clip((1.0 - gripper_cmd) * 255.0, 0.0, 255.0)
        mujoco.mj_step(self.model, self.data, nstep=50)
        mujoco.mj_forward(self.model, self.data)

        self._step_count += 1
        self._check_grasp()
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._success
        truncated = self._step_count >= self.MAX_STEPS

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

    def _randomize_cube(self):
        """Randomize cube position within workspace."""
        x = np.random.uniform(0.35, 0.55)
        y = np.random.uniform(-0.30, 0.30)
        self.data.qpos[
            self.model.jnt_qposadr[
                self.model.body_jntadr[self._cube_body_id]
            ]:self.model.jnt_qposadr[
                self.model.body_jntadr[self._cube_body_id]
            ] + 3
        ] = [x, y, self.CUBE_HALF]

    def _get_obs(self) -> np.ndarray:
        """14D state vector."""
        return np.concatenate([
            self.joint_positions,      # 7
            self.ee_position,           # 3
            self.cube_position,         # 3
            [self.gripper_width],       # 1
        ]).astype(np.float32)

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
        """Check if cube is grasped (between fingers)."""
        dist = np.linalg.norm(self.ee_position[:2] - self.cube_position[:2])
        z_diff = abs(self.ee_position[2] - self.cube_position[2] - self.CUBE_HALF)
        if dist < self.GRASP_THRESH and z_diff < 0.02 and self.gripper_width < 0.02:
            self._grasped = True
        if self._grasped and self.cube_position[2] > self.SUCCESS_Z:
            self._success = True

    def _compute_reward(self) -> float:
        """Dense reward: approach + grasp + lift."""
        ee = self.ee_position
        cube = self.cube_position

        # Distance reward (negative L2, scaled)
        xy_dist = np.linalg.norm(ee[:2] - cube[:2])
        z_dist = abs(ee[2] - (cube[2] + self.CUBE_HALF + 0.01))
        r_approach = -xy_dist - 0.5 * z_dist

        # Grasp bonus
        r_grasp = 0.0
        if self._grasped and not getattr(self, '_grasp_bonus_given', False):
            r_grasp = 5.0
            self._grasp_bonus_given = True

        # Success bonus
        r_success = 0.0
        if self._success and not getattr(self, '_success_bonus_given', False):
            r_success = 20.0
            self._success_bonus_given = True

        # Gripper penalty: encourage closing near cube
        r_grip = 0.0
        if xy_dist < self.GRASP_THRESH * 2:
            r_grip = 0.1 * (1.0 - self.gripper_width / 0.08)  # reward closing

        # Step penalty to encourage speed
        r_step = -0.01

        return r_approach + r_grasp + r_success + r_grip + r_step

    def close(self):
        pass
