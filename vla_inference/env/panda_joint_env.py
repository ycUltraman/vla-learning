"""Joint-control Franka Panda MuJoCo environment for PI0.5 inference.

Matches the sim-teleop dataset observation/action format:
- Action: 8D (7 joint position targets + gripper command 0-1)
- State: 15D (7 joint pos + gripper width + EE xyz + EE quat wxyz)
- Cameras: rendered via MjvCamera objects (not XML-defined)

Loads mujoco_menagerie's scene.xml directly to avoid mesh path issues.
"""

import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
from typing import Optional

# Paths to available scenes
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_MENAGERIE_SCENE = str(
    _PROJECT_ROOT / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"
)
_TASK_SCENE = str(
    _PROJECT_ROOT / "mujoco_menagerie" / "franka_emika_panda" / "scene_task.xml"
)


class PandaJointEnv:
    """
    Minimal joint-control MuJoCo environment for PI0.5 policy inference.

    Loads the menagerie Franka Panda scene and uses MjvCamera objects
    for on-demand image rendering, avoiding XML include path issues.
    """

    HOME_QPOS = np.array(
        [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04]
    )
    HOME_CTRL = np.array(
        [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 255.0]
    )

    EE_OFFSET = np.array([0.0, 0.0, 0.15])  # hand center → gripper tip in hand local Z
    CAM_W, CAM_H = 640, 480

    def __init__(self, render_mode: str = "human", scene: str = "task"):
        xml_path = _TASK_SCENE if scene == "task" else _MENAGERIE_SCENE
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        self._resolve_ids()

        # Shared renderer
        self._renderer = mujoco.Renderer(self.model, self.CAM_H, self.CAM_W)

        # Front camera — "desk operation" viewpoint matching sim-teleop
        # Robot should occupy ~50% of frame, cubes ~30-50px, tray clearly visible
        self._cam_front = mujoco.MjvCamera()
        self._cam_front.type = mujoco.mjtCamera.mjCAMERA_FREE
        # Camera in FRONT of robot looking at workspace
        self._cam_front.lookat = np.array([0.6, -0.1, 0.32])
        self._cam_front.distance = 0.50
        self._cam_front.azimuth = 180
        self._cam_front.elevation = -25.0

        # Wrist camera: XML-defined, follows hand via targetbody mode
        self._wrist_cam_name = "wrist_cam"

        self.viewer = None
        if render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.reset()

    def close(self):
        self._renderer.close()
        if self.viewer is not None:
            self.viewer.close()

    # ── Observation accessors ───────────────────────────────────

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
    def ee_quaternion(self) -> np.ndarray:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.xmat[self._hand_body_id])
        return quat

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

    def _get_obs(self) -> dict:
        state = np.concatenate([
            self.joint_positions,
            [self.gripper_width],
            self.ee_position,
            self.ee_quaternion,
        ]).astype(np.float32)

        return {
            "observation.state": state,
            "observation.images.front": self._render_view(self._cam_front),
            "observation.images.wrist": self._render_view(self._wrist_cam_name),
        }

    def _render_view(self, cam) -> np.ndarray:
        """Render one camera view (MjvCamera or XML camera name)."""
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()

    # def _update_wrist_camera(self):

    #     hand_pos = self.data.xpos[self._hand_body_id]

    #     self._cam_wrist.lookat = hand_pos + np.array([0,0,-0.10])
    #     self._cam_wrist.distance = 0.3
    #     self._cam_wrist.azimuth = 90
    #     self._cam_wrist.elevation = -60

    def step(self, action: np.ndarray) -> dict:
        """action: [joint1..joint7, gripper_cmd]  gripper_cmd: 0=open 1=closed"""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (8,):
            raise ValueError(f"Expected (8,), got {action.shape}")

        self.data.ctrl[:7] = action[:7]
        self.data.ctrl[7] = np.clip((1.0 - action[7]) * 255.0, 0.0, 255.0)

        mujoco.mj_step(self.model, self.data, nstep=50)
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        if self.viewer is not None:
            self.viewer.sync()
        return obs

    def compute_target_joints(self, ee_delta: np.ndarray, damped: bool = True) -> np.ndarray:
        """Compute new persistent joint targets from EE delta. Does NOT step.

        6D Jacobian: position + orientation hold (rot_error=0).
        damped=True: 50% step (for stable teleop).
        damped=False: full step (for replay/inference fidelity).

        Returns: (7,) new target joint positions.
        """
        ee_delta = np.asarray(ee_delta, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, self._hand_body_id)
        Jp = jac_pos[:, :7]
        Jr = jac_rot[:, :7]

        pos_error = ee_delta
        rot_error = np.zeros(3)

        error = np.concatenate([pos_error, rot_error])
        J = np.vstack([Jp, Jr])

        damping = 0.1
        dq = J.T @ np.linalg.inv(J @ J.T + damping * damping * np.eye(6)) @ error
        if damped:
            dq *= 0.5
        self._ee_target_joints[:7] += dq

        for i in range(7):
            lo = self.model.jnt_range[i][0]
            hi = self.model.jnt_range[i][1]
            self._ee_target_joints[i] = np.clip(self._ee_target_joints[i], lo, hi)

        return self._ee_target_joints.copy()

    def apply_ee_delta(self, ee_action: np.ndarray) -> dict:
        """Apply EE delta action [dx,dy,dz,drx,dry,drz,gripper] (7D).

        3 IK+PD cycles with undamped steps — matches teleop pattern,
        verified at EE error mean=0.024.
        """
        ee_delta_total = ee_action[:3]
        gripper_cmd = ee_action[6]
        obs = None
        for i in range(3):
            self.compute_target_joints(ee_delta_total / 3, damped=False)
            obs = self.step(np.concatenate([self._ee_target_joints, [gripper_cmd]]))
        return obs

    def step_ee(self, ee_delta: np.ndarray, gripper_cmd: float) -> dict:
        """Cartesian control — 6D damped least-squares IK.

        Position: moves EE by ee_delta.
        Orientation: holds current rotation (rot_error = 0).
        Uses all 7 joints via damped pseudo-inverse.

        Args:
            ee_delta: (3,) EE position delta (meters)
            gripper_cmd: 0=open, 1=closed
        """
        ee_delta = np.asarray(ee_delta, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

        # Current pose
        current_pos = self.ee_position.copy()
        target_pos = current_pos + ee_delta

        # Jacobian
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, self._hand_body_id)
        Jp = jac_pos[:, :7]  # (3, 7)
        Jr = jac_rot[:, :7]  # (3, 7)

        # Position error
        pos_error = target_pos - current_pos

        # Orientation error: keep current (zero desired change)
        rot_error = np.zeros(3)

        error = np.concatenate([pos_error, rot_error])  # (6,)
        J = np.vstack([Jp, Jr])  # (6, 7)

        # Damped least squares
        damping = 0.02
        dq = J.T @ np.linalg.inv(J @ J.T + damping * damping * np.eye(6)) @ error
        dq *= 0.5  # smaller step for stability

        self._ee_target_joints[:7] += dq

        # Joint limits
        for i in range(7):
            lo = self.model.jnt_range[i][0]
            hi = self.model.jnt_range[i][1]
            self._ee_target_joints[i] = np.clip(self._ee_target_joints[i], lo, hi)

        self.data.ctrl[7] = np.clip((1.0 - gripper_cmd) * 255.0, 0.0, 255.0)

        return self.step(np.concatenate([self._ee_target_joints, [gripper_cmd]]))

    def get_action_from_ee(self, gripper_cmd: float) -> np.ndarray:
        """After EE step, read joint positions as 8D action for recording."""
        return np.concatenate([self.joint_positions, [gripper_cmd]])

    def reset(self) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        n_robot = len(self.HOME_QPOS)
        self.data.qpos[:n_robot] = self.HOME_QPOS
        mujoco.mj_forward(self.model, self.data)
        # Persistent EE target joints + home orientation for 6D IK
        self._ee_target_joints = self.joint_positions.copy()
        self._home_ee_quat = self.ee_quaternion.copy()
        return self._get_obs()


def _quat_conj(q):
    """Quaternion conjugate: (w,x,y,z) → (w,-x,-y,-z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])
