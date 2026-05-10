# Two ways to make the robot stand:
#   1. Write your own controller using standing_ctrl_template.py, rename it to standing_ctrl.py
#   2. Tune configs/scene27_cue.yaml so this controller works

import mujoco
import numpy as np
import cvxpy as cp
import yaml
import os
from typing import Tuple, Optional, Dict


class LagrangianDynamics:

    def __init__(self, xml_path: str,
                 left_foot_body: str = 'left_ankle_roll_link',
                 right_foot_body: str = 'right_ankle_roll_link'):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.nv = self.model.nv
        self.nq = self.model.nq

        self.left_foot_body = left_foot_body
        self.right_foot_body = right_foot_body

        self._M = np.zeros((self.nv, self.nv))
        self._h_full = np.zeros(self.nv)
        self._h_gravity = np.zeros(self.nv)

        self._qvel_filt = None
        self._last_w = None
        self._qp_problem = None
        self._qp_vars = None
        self._qp_params = {}

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        assert qpos.shape[0] == self.nq, f"qpos must have shape ({self.nq},), got {qpos.shape}"
        assert qvel.shape[0] == self.nv, f"qvel must have shape ({self.nv},), got {qvel.shape}"

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def _get_body_jacobians(self, body_name: str) -> Tuple[int, np.ndarray, np.ndarray]:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid == -1:
            raise ValueError(f"Body '{body_name}' not found in model.")

        Jp = np.zeros((3, self.nv))
        Jr = np.zeros((3, self.nv))
        mujoco.mj_jacBody(self.model, self.data, Jp, Jr, bid)

        return bid, Jp, Jr

    def _get_body_rotation_world_to_body(self, bid: int) -> np.ndarray:
        R_wb = self.data.xmat[bid].reshape(3, 3).copy()
        R_bw = R_wb.T
        return R_bw

    def compute_mass_matrix(self) -> np.ndarray:
        mujoco.mj_fullM(self.model, self._M, self.data.qM)
        return self._M.copy()

    def compute_bias_vector(self,
                            use_velocity_terms: bool = True,
                            vel_alpha: float = 0.2,
                            coriolis_scale_ank: float = 0.2,
                            coriolis_clip_Nm: float = 10.0) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)

        qv_meas = self.data.qvel.copy()
        if self._qvel_filt is None:
            self._qvel_filt = qv_meas.copy()
        qv_filt = (1.0 - vel_alpha) * self._qvel_filt + vel_alpha * qv_meas
        self._qvel_filt = qv_filt.copy()

        qvel_backup = self.data.qvel.copy()

        if use_velocity_terms:
            self.data.qvel[:] = qv_filt
        else:
            self.data.qvel[:] = 0.0

        self.data.qacc[:] = 0.0
        h_CG = np.empty(self.nv, dtype=float)
        mujoco.mj_rne(self.model, self.data, 0, h_CG)

        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        h_G = np.empty(self.nv, dtype=float)
        mujoco.mj_rne(self.model, self.data, 0, h_G)

        self.data.qvel[:] = qvel_backup

        h_C = h_CG - h_G

        ankle_joints = ['left_ankle_pitch', 'left_ankle_roll',
                        'right_ankle_pitch', 'right_ankle_roll']

        for joint_name in ankle_joints:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jnt_id >= 0:
                j = self.model.jnt_dofadr[jnt_id]
                h_C[j] *= coriolis_scale_ank
                if coriolis_clip_Nm > 0:
                    h_C[j] = np.clip(h_C[j], -coriolis_clip_Nm, coriolis_clip_Nm)

        h = h_G + h_C
        return h

    def compute_gravity_vector(self) -> np.ndarray:
        qvel_backup = self.data.qvel.copy()
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        mujoco.mj_rne(self.model, self.data, 0, self._h_gravity)
        self.data.qvel[:] = qvel_backup
        return self._h_gravity.copy()

    def compute_MCG(self,
                    qpos: np.ndarray,
                    qvel: np.ndarray,
                    use_velocity_terms: bool = True,
                    mu: float = 0.6,
                    cop_x_max: float = 0.06,
                    cop_y_max: float = 0.03,
                    torsion_ratio: float = 0.02,
                    w_base: float = 1.0,
                    w_total: float = 1e6,
                    w_split: float = 1e4,
                    w_tan: float = 5e2,
                    w_yaw: float = 1e2,
                    w_reg: float = 1e-6,
                    eta_total: float = 0.0,
                    gamma: float = 0.5,
                    vel_alpha: float = 0.2,
                    coriolis_scale_ank: float = 0.2,
                    coriolis_clip_Nm: float = 10.0,
                    w_smooth: float = 8e2,
                    solver: str = 'OSQP',
                    verbose: bool = False
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.set_state(qpos, qvel)

        M = self.compute_mass_matrix()

        h = self.compute_bias_vector(
            use_velocity_terms=use_velocity_terms,
            vel_alpha=vel_alpha,
            coriolis_scale_ank=coriolis_scale_ank,
            coriolis_clip_Nm=coriolis_clip_Nm
        )

        h_base = h[:6]
        h_joints = h[6:]

        bidL, JpL, JrL = self._get_body_jacobians(self.left_foot_body)
        bidR, JpR, JrR = self._get_body_jacobians(self.right_foot_body)

        JpL_b, JpL_j = JpL[:, :6], JpL[:, 6:]
        JrL_b, JrL_j = JrL[:, :6], JrL[:, 6:]
        JpR_b, JpR_j = JpR[:, :6], JpR[:, 6:]
        JrR_b, JrR_j = JrR[:, :6], JrR[:, 6:]

        Rbw_L = self._get_body_rotation_world_to_body(bidL)
        Rbw_R = self._get_body_rotation_world_to_body(bidR)

        H_b = np.zeros((6, 12))
        H_b[:, 0:3] = JpL_b.T
        H_b[:, 3:6] = JrL_b.T
        H_b[:, 6:9] = JpR_b.T
        H_b[:, 9:12] = JrR_b.T

        total_mass = float(np.sum(self.model.body_mass))
        g = float(self.model.opt.gravity[2])
        Fz_target = (1.0 + eta_total) * total_mass * (-g)

        T_L = np.zeros((6, 12))
        T_R = np.zeros((6, 12))
        T_L[0:3, 0:3] = Rbw_L
        T_L[3:6, 3:6] = Rbw_L
        T_R[0:3, 6:9] = Rbw_R
        T_R[3:6, 9:12] = Rbw_R

        # Build the QP problem once with cp.Parameter objects; subsequent calls
        # only update parameter values and re-solve, avoiding CVXPY recompilation.
        if self._qp_problem is None:
            w_var = cp.Variable(12)
            p_H_b = cp.Parameter((6, 12))
            p_h_base = cp.Parameter(6)
            p_T_L = cp.Parameter((6, 12))
            p_T_R = cp.Parameter((6, 12))
            p_last_w = cp.Parameter(12)

            wL_b = p_T_L @ w_var
            wR_b = p_T_R @ w_var
            fxL_b, fyL_b, fzL_b = wL_b[0], wL_b[1], wL_b[2]
            mxL_b, myL_b, mzL_b = wL_b[3], wL_b[4], wL_b[5]
            fxR_b, fyR_b, fzR_b = wR_b[0], wR_b[1], wR_b[2]
            mxR_b, myR_b, mzR_b = wR_b[3], wR_b[4], wR_b[5]

            t_sel = np.zeros(12); t_sel[2] = 1.0; t_sel[8] = 1.0
            s_sel = np.zeros(12); s_sel[2] = (1.0 - gamma); s_sel[8] = -gamma

            obj_terms = [
                w_base  * cp.sum_squares(p_H_b @ w_var - p_h_base),
                w_total * cp.sum_squares(t_sel @ w_var - Fz_target),
                w_split * cp.sum_squares(s_sel @ w_var),
                w_tan   * cp.sum_squares(cp.hstack([w_var[i] for i in [0, 1, 6, 7]])),
                w_yaw   * cp.sum_squares(cp.hstack([w_var[i] for i in [5, 11]])),
                w_reg   * cp.sum_squares(w_var),
                w_smooth * cp.sum_squares(w_var - p_last_w),
            ]
            objective = cp.Minimize(cp.sum(obj_terms))

            cons = [
                fzL_b >= 0,
                fxL_b <= mu * fzL_b,  -fxL_b <= mu * fzL_b,
                fyL_b <= mu * fzL_b,  -fyL_b <= mu * fzL_b,
                mxL_b <= cop_y_max * fzL_b,  -mxL_b <= cop_y_max * fzL_b,
                myL_b <= cop_x_max * fzL_b,  -myL_b <= cop_x_max * fzL_b,
                mzL_b <= torsion_ratio * fzL_b,  -mzL_b <= torsion_ratio * fzL_b,
                fzR_b >= 0,
                fxR_b <= mu * fzR_b,  -fxR_b <= mu * fzR_b,
                fyR_b <= mu * fzR_b,  -fyR_b <= mu * fzR_b,
                mxR_b <= cop_y_max * fzR_b,  -mxR_b <= cop_y_max * fzR_b,
                myR_b <= cop_x_max * fzR_b,  -myR_b <= cop_x_max * fzR_b,
                mzR_b <= torsion_ratio * fzR_b,  -mzR_b <= torsion_ratio * fzR_b,
            ]

            self._qp_vars = w_var
            self._qp_params = {
                'H_b': p_H_b, 'h_base': p_h_base,
                'T_L': p_T_L, 'T_R': p_T_R,
                'last_w': p_last_w,
            }
            self._qp_problem = cp.Problem(objective, cons)

        # Update parameter values for this call
        self._qp_params['H_b'].value = H_b
        self._qp_params['h_base'].value = h_base
        self._qp_params['T_L'].value = T_L
        self._qp_params['T_R'].value = T_R
        self._qp_params['last_w'].value = (
            self._last_w.copy() if self._last_w is not None else np.zeros(12)
        )
        w_var = self._qp_vars
        if self._last_w is not None:
            w_var.value = self._last_w.copy()

        solve_kwargs = dict(verbose=verbose, warm_start=True)
        if solver.upper() == 'OSQP':
            solve_kwargs.update(dict(eps_abs=1e-8, eps_rel=1e-8, max_iter=100000))

        self._qp_problem.solve(solver=getattr(cp, solver), **solve_kwargs)

        if w_var.value is None:
            raise RuntimeError(f"QP failed with status {self._qp_problem.status}")

        w_world = w_var.value.astype(float)
        self._last_w = w_world.copy()

        wL = w_world[0:6].copy()
        wR = w_world[6:12].copy()

        fL_w, mL_w = wL[0:3], wL[3:6]
        fR_w, mR_w = wR[0:3], wR[3:6]

        tau_joints = h_joints.copy()
        tau_joints -= (JpL_j.T @ fL_w + JrL_j.T @ mL_w)
        tau_joints -= (JpR_j.T @ fR_w + JrR_j.T @ mR_w)

        return tau_joints, M, h, wL, wR

    def get_joint_info(self) -> Dict:
        joints = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                joints[name] = {
                    'id': i,
                    'type': self.model.jnt_type[i],
                    'dof_addr': self.model.jnt_dofadr[i],
                    'qpos_addr': self.model.jnt_qposadr[i],
                }
        return joints

    @property
    def total_mass(self) -> float:
        return float(np.sum(self.model.body_mass))

    @property
    def gravity_vector(self) -> np.ndarray:
        return self.model.opt.gravity.copy()

    def reset_filters(self) -> None:
        self._qvel_filt = None
        self._last_w = None

class StandingCtrl:

    def __init__(self,
                 config_path: str,
                 pelvis_body: str = 'pelvis',
                 left_foot_body: str = 'left_ankle_roll_link',
                 right_foot_body: str = 'right_ankle_roll_link'):
        self.config_path = config_path
        config_abs_path = os.path.abspath(config_path)
        self.config_dir = os.path.dirname(os.path.dirname(config_abs_path))
        self.config = self._load_config(config_path)

        xml_path = self.config['xml_path'].replace('{DIR}', self.config_dir)

        self.simulation_dt = self.config.get('simulation_dt', 0.002)
        self.control_decimation = self.config.get('control_decimation', 10)
        self.control_dt = self.config.get('control_dt', self.simulation_dt * self.control_decimation)
        self.num_joints = self.config.get('num_actions', 12)

        self.arm_waist_kps = np.array(self.config['arm_waist_kps'], dtype=np.float64)
        self.arm_waist_kds = np.array(self.config['arm_waist_kds'], dtype=np.float64)
        self.arm_waist_target = np.array(self.config['arm_waist_target'], dtype=np.float64)
        self.num_arm_joints = len(self.arm_waist_target)
        self.num_actuators = self.num_joints + self.num_arm_joints
        self.arm_qpos_start = 7 + self.num_joints
        self.arm_qvel_start = 6 + self.num_joints

        self.dynamics = LagrangianDynamics(
            xml_path,
            left_foot_body=left_foot_body,
            right_foot_body=right_foot_body
        )

        self.model = self.dynamics.model
        self.data = self.dynamics.data
        self.nv = self.dynamics.nv
        self.nq = self.dynamics.nq

        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, pelvis_body)
        self.left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, left_foot_body)
        self.right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, right_foot_body)

        if self.pelvis_id == -1:
            raise ValueError(f"Pelvis body '{pelvis_body}' not found")

        self._jac_l = np.zeros((6, self.nv))
        self._jac_r = np.zeros((6, self.nv))

        self._prev_torque = np.zeros(self.num_joints)
        self._prev_com = None
        self._com_error_integral = np.zeros(3)
        self._base_xy_integral = np.zeros(2)
        self._prev_roll = None
        self._prev_pitch = None
        self._prev_base_xy = None

        self._qpos_filtered = None
        self._qvel_filtered = None
        self.qpos_filter_alpha = float(self.config.get('qpos_filter_alpha', 0.2))
        self.qvel_filter_alpha = float(self.config.get('qvel_filter_alpha', 0.15))
        self.com_vel_filter_alpha = float(self.config.get('com_vel_filter_alpha', 0.15))

        self.stand_kps_tau = np.array(self.config.get('stand_kps_tau', [0] * self.num_joints), dtype=np.float32)
        self.stand_kds_tau = np.array(self.config.get('stand_kds_tau', [0] * self.num_joints), dtype=np.float32)
        self.full_stand_kps_tau = np.array(self.config.get('full_stand_kps_tau', [0] * self.num_joints), dtype=np.float32)
        self.full_stand_kds_tau = np.array(self.config.get('full_stand_kds_tau', [0] * self.num_joints), dtype=np.float32)
        self.com_kps = np.array(self.config.get('CoM_kps', [0, 0, 0]), dtype=np.float32)
        self.com_kis = np.array(self.config.get('CoM_kis', [0, 0, 0]), dtype=np.float32)
        self.com_kds = np.array(self.config.get('CoM_kds', [0, 0, 0]), dtype=np.float32)
        self.com_max_force = float(self.config.get('CoM_max_force', 500.0))
        self.com_max_integral = float(self.config.get('CoM_max_integral', 0.5))

        self.hip_roll_scale = float(self.config.get('hip_roll_scale', 0.0))
        self.hip_pitch_scale = float(self.config.get('hip_pitch_scale', 0.0))
        self.hip_roll_lat_scale = float(self.config.get('hip_roll_lat_scale', 0.0))
        self.ankle_pitch_scale = float(self.config.get('ankle_pitch_scale', 1.0))
        self.ankle_roll_scale = float(self.config.get('ankle_roll_scale', 1.0))
        self.base_recenter_kp = np.array(
            self.config.get('base_recenter_kp', [200.0, 200.0]),
            dtype=np.float64
        )
        self.base_recenter_kd = np.array(
            self.config.get('base_recenter_kd', [120.0, 120.0]),
            dtype=np.float64
        )
        self.body_roll_kp = float(self.config.get('body_roll_kp', 70.0))
        self.body_roll_kd = float(self.config.get('body_roll_kd', 14.0))
        self.body_pitch_kp = float(self.config.get('body_pitch_kp', 85.0))
        self.body_pitch_kd = float(self.config.get('body_pitch_kd', 16.0))
        self.body_roll_ankle_scale = float(self.config.get('body_roll_ankle_scale', 0.9))
        self.body_roll_hip_scale = float(self.config.get('body_roll_hip_scale', 0.35))
        self.body_pitch_ankle_scale = float(self.config.get('body_pitch_ankle_scale', 1.0))
        self.body_pitch_hip_scale = float(self.config.get('body_pitch_hip_scale', 0.3))
        self.base_drift_alpha = np.array(
            self.config.get('base_drift_alpha', [0.0, 0.0]),
            dtype=np.float64
        )
        self.base_drift_deadband = np.array(
            self.config.get('base_drift_deadband', [0.0, 0.0]),
            dtype=np.float64
        )
        self.base_xy_integral_k = np.array(
            self.config.get('base_xy_integral_k', [0.0, 0.0]),
            dtype=np.float64
        )
        self.base_xy_integral_max = np.array(
            self.config.get('base_xy_integral_max', [0.0, 0.0]),
            dtype=np.float64
        )
        self.base_xy_integral_deadband = np.array(
            self.config.get('base_xy_integral_deadband', [0.0, 0.0]),
            dtype=np.float64
        )
        self.base_xy_damping_k = np.array(
            self.config.get('base_xy_damping_k', [0.0, 0.0]),
            dtype=np.float64
        )
        self.base_xy_damping_deadband = np.array(
            self.config.get('base_xy_damping_deadband', [0.0, 0.0]),
            dtype=np.float64
        )

        self.torque_limits = np.array(
            self.config.get('stand_torque_limits', [100] * self.num_joints),
            dtype=np.float32
        )
        self.delta_torque = self.torque_limits * float(self.config.get('delta_torque_scale', 1.5))

        self.standing_angles = np.array(
            self.config.get('standing_angles', [0] * self.num_joints),
            dtype=np.float32
        )

        self.stand_CoM = np.array(self.config.get('stand_CoM', [0, 0, 1.0]), dtype=np.float32)

        self.dof_pos_lower_limits = np.array(
            self.config.get('dof_pos_lower_limits', [-np.pi] * self.num_joints),
            dtype=np.float32
        )
        self.dof_pos_upper_limits = np.array(
            self.config.get('dof_pos_upper_limits', [np.pi] * self.num_joints),
            dtype=np.float32
        )

        qp_config = self.config.get('qp_params', {})
        self.qp_params = {
            'mu': float(qp_config.get('mu', 0.5)),
            'cop_x_max': float(qp_config.get('cop_x_max', 0.15)),
            'cop_y_max': float(qp_config.get('cop_y_max', 0.035)),
            'torsion_ratio': float(qp_config.get('torsion_ratio', 0.02)),
            'w_base': float(qp_config.get('w_base', 1e6)),
            'w_total': float(qp_config.get('w_total', 1e6)),
            'w_split': float(qp_config.get('w_split', 1e3)),
            'w_tan': float(qp_config.get('w_tan', 5e2)),
            'w_yaw': float(qp_config.get('w_yaw', 1e2)),
            'w_reg': float(qp_config.get('w_reg', 1e-6)),
            'w_smooth': float(qp_config.get('w_smooth', 8e2)),
            'gamma': float(qp_config.get('gamma', 0.5)),
            'vel_alpha': float(qp_config.get('vel_alpha', 0.2)),
            'coriolis_scale_ank': float(qp_config.get('coriolis_scale_ank', 0.2)),
            'coriolis_clip_Nm': float(qp_config.get('coriolis_clip_Nm', 10.0)),
        }

        self._standing_yaw_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._foot_world_pos_standing = self._compute_foot_world_positions_standing()
        self._standing_world_com = self._compute_standing_world_com()
        self._base_target_xy = self.stand_CoM[:2].copy()

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def get_xml_path(self) -> str:
        return self.config['xml_path'].replace('{DIR}', self.config_dir)

    def get_duration(self) -> float:
        return self.config.get('simulation_duration', 30)

    def get_initial_state(self) -> dict:
        qpos = np.zeros(self.nq)
        qpos[:3] = self.stand_CoM

        qpos[3:7] = [1, 0, 0, 0]
        qpos[7:7 + self.num_joints] = self.standing_angles
        qpos[self.arm_qpos_start:self.arm_qpos_start + self.num_arm_joints] = self.arm_waist_target
        qvel = np.zeros(self.nv)
        return {'qpos': qpos, 'qvel': qvel}

    def _filter_state(self, qpos: np.ndarray, qvel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._qpos_filtered is None:
            self._qpos_filtered = qpos.copy()
            self._qvel_filtered = qvel.copy()
            return qpos.copy(), qvel.copy()

        self._qpos_filtered = (self.qpos_filter_alpha * qpos +
                               (1 - self.qpos_filter_alpha) * self._qpos_filtered)
        self._qvel_filtered = (self.qvel_filter_alpha * qvel +
                               (1 - self.qvel_filter_alpha) * self._qvel_filtered)

        return self._qpos_filtered.copy(), self._qvel_filtered.copy()

    def _yaw_from_quat(self, quat: np.ndarray) -> float:
        w, x, y, z = quat
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _yaw_quat_from_quat(self, quat: np.ndarray) -> np.ndarray:
        yaw = self._yaw_from_quat(quat)
        half = 0.5 * yaw
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)

    def _world_xy_to_standing_local(self, xy: np.ndarray) -> np.ndarray:
        """Rotate world XY vector components into the current standing yaw frame."""
        yaw = self._yaw_from_quat(self._standing_yaw_quat)
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([c * xy[0] + s * xy[1], -s * xy[0] + c * xy[1]], dtype=np.float64)

    def _standing_local_xy_to_world(self, xy: np.ndarray) -> np.ndarray:
        """Rotate standing-frame XY vector components back into world coordinates."""
        yaw = self._yaw_from_quat(self._standing_yaw_quat)
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]], dtype=np.float64)

    def _quat_conjugate(self, quat: np.ndarray) -> np.ndarray:
        return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)

    def _quat_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=np.float64,
        )

    def _quat_to_roll_pitch_in_standing_frame(self, quat: np.ndarray) -> Tuple[float, float]:
        rel_quat = self._quat_multiply(self._quat_conjugate(self._standing_yaw_quat), quat)
        return self._quat_to_roll_pitch(rel_quat)

    def update_standing_reference_from_state(self, qpos: np.ndarray) -> None:
        """Rebuild world-frame standing references for the current base XY/yaw.

        The controller was originally tuned at the origin with identity yaw.  For
        teleported pool shots, the same local standing pose is placed elsewhere
        and rotated.  The support-foot and target-CoM references must be rebuilt
        in that new world frame; otherwise the controller pushes as if the robot
        should still be standing at the original yaw.
        """
        self.stand_CoM[:2] = qpos[:2]
        self._base_target_xy = np.array(qpos[:2], dtype=np.float64)
        self._standing_yaw_quat = self._yaw_quat_from_quat(qpos[3:7])
        self._foot_world_pos_standing = self._compute_foot_world_positions_standing()
        self._standing_world_com = self._compute_standing_world_com()

    def _compute_foot_world_positions_standing(self) -> Dict[str, np.ndarray]:
        self.data.qpos[:3] = self.stand_CoM
        self.data.qpos[3:7] = self._standing_yaw_quat
        self.data.qpos[7:7 + self.num_joints] = self.standing_angles
        self.data.qpos[self.arm_qpos_start:self.arm_qpos_start + self.num_arm_joints] = self.arm_waist_target
        mujoco.mj_forward(self.model, self.data)

        self._foot_standing_rot = {
            'left': self.data.xmat[self.left_foot_id].reshape(3, 3).copy(),
            'right': self.data.xmat[self.right_foot_id].reshape(3, 3).copy()
        }

        return {
            'left': self.data.xpos[self.left_foot_id].copy(),
            'right': self.data.xpos[self.right_foot_id].copy()
        }

    def _compute_standing_world_com(self) -> np.ndarray:
        self.data.qpos[:3] = self.stand_CoM
        self.data.qpos[3:7] = self._standing_yaw_quat
        self.data.qpos[7:7 + self.num_joints] = self.standing_angles
        self.data.qpos[self.arm_qpos_start:self.arm_qpos_start + self.num_arm_joints] = self.arm_waist_target
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.data.subtree_com[self.pelvis_id].copy()

    def _quat_to_roll_pitch(self, quat: np.ndarray) -> Tuple[float, float]:
        w, x, y, z = quat

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        return roll, pitch

    def _determine_stance_foot(self, joint_angles: np.ndarray) -> str:
        self.data.qpos[:3] = np.array([0, 0, 2])
        self.data.qpos[3:7] = np.array([1, 0, 0, 0])
        self.data.qpos[7:7 + self.num_joints] = joint_angles
        mujoco.mj_forward(self.model, self.data)

        left_z = self.data.xpos[self.left_foot_id][2]
        right_z = self.data.xpos[self.right_foot_id][2]

        return 'left' if left_z <= right_z else 'right'

    def set_gains(self,
                  stand_kps: Optional[np.ndarray] = None,
                  stand_kds: Optional[np.ndarray] = None,
                  com_kps: Optional[np.ndarray] = None,
                  com_kds: Optional[np.ndarray] = None) -> None:
        if stand_kps is not None:
            self.stand_kps = np.asarray(stand_kps, dtype=np.float32)
        if stand_kds is not None:
            self.stand_kds = np.asarray(stand_kds, dtype=np.float32)
        if com_kps is not None:
            self.com_kps = np.asarray(com_kps, dtype=np.float32)
        if com_kds is not None:
            self.com_kds = np.asarray(com_kds, dtype=np.float32)

    def set_torque_limits(self,
                          torque_limits: np.ndarray,
                          delta_scale: float = 1.5) -> None:
        self.torque_limits = np.asarray(torque_limits, dtype=np.float32)
        self.delta_torque = self.torque_limits * delta_scale

    def set_standing_pose(self, standing_angles: np.ndarray) -> None:
        self.standing_angles = np.asarray(standing_angles, dtype=np.float32)

    def _get_com_position(self, joint_angles: np.ndarray, stance_foot: str,
                          base_quat: np.ndarray = None) -> np.ndarray:
        self.data.qpos[:3] = np.array([0, 0, 2])
        if base_quat is not None:
            self.data.qpos[3:7] = base_quat
        else:
            self.data.qpos[3:7] = np.array([1, 0, 0, 0])
        self.data.qpos[7:7 + self.num_joints] = joint_angles
        self.data.qpos[self.arm_qpos_start:self.arm_qpos_start + self.num_arm_joints] = self.arm_waist_target
        mujoco.mj_forward(self.model, self.data)

        com_pos = self.data.subtree_com[self.pelvis_id].copy()

        if stance_foot == 'left':
            foot_id = self.left_foot_id
            foot_world_pos = self._foot_world_pos_standing['left']
        else:
            foot_id = self.right_foot_id
            foot_world_pos = self._foot_world_pos_standing['right']

        foot_pos = self.data.xpos[foot_id].copy()

        diff_global = com_pos - foot_pos

        result = foot_world_pos + diff_global

        return result

    def _com_pid_control(self,
                        target_com: np.ndarray,
                        current_com: np.ndarray,
                        current_com_vel: np.ndarray) -> np.ndarray:
        error = target_com - current_com

        self._com_error_integral += error * self.control_dt
        self._com_error_integral = np.clip(self._com_error_integral,
                                           -self.com_max_integral, self.com_max_integral)

        F_com = (self.com_kps * error
                 + self.com_kis * self._com_error_integral
                 - self.com_kds * current_com_vel)
        return np.clip(F_com, -self.com_max_force, self.com_max_force)

    def _apply_torque_limits(self, tau: np.ndarray) -> np.ndarray:
        delta_tau = tau - self._prev_torque
        clip_delta_tau = np.clip(delta_tau, -self.delta_torque, self.delta_torque)
        commanded_tau = self._prev_torque + clip_delta_tau

        final_tau = np.clip(commanded_tau, -self.torque_limits, self.torque_limits)

        self._prev_torque = final_tau.copy()

        return final_tau

    def compute_torque(self,
                       qpos: np.ndarray,
                       qvel: np.ndarray) -> Tuple[np.ndarray, Dict]:
        info = {}

        raw_qpos = qpos.copy()
        raw_qvel = qvel.copy()

        qpos, qvel = self._filter_state(qpos, qvel)

        qpos_standing = qpos.copy()
        qpos_standing[:3] = self.stand_CoM
        qpos_standing[3:7] = self._standing_yaw_quat
        qpos_standing[7:7 + self.num_joints] = self.standing_angles
        qvel_zero = np.zeros_like(qvel)

        try:
            qp = self.qp_params
            tau_cg, M_standing, h, wL, wR = self.dynamics.compute_MCG(
                qpos_standing, qvel_zero,
                use_velocity_terms=False,
                mu=qp['mu'],
                cop_x_max=qp['cop_x_max'],
                cop_y_max=qp['cop_y_max'],
                torsion_ratio=qp['torsion_ratio'],
                w_base=qp['w_base'],
                w_total=qp['w_total'],
                w_split=qp['w_split'],
                w_tan=qp['w_tan'],
                w_yaw=qp['w_yaw'],
                w_reg=qp['w_reg'],
                gamma=qp['gamma'],
                vel_alpha=qp['vel_alpha'],
                coriolis_scale_ank=qp['coriolis_scale_ank'],
                coriolis_clip_Nm=qp['coriolis_clip_Nm'],
                w_smooth=qp['w_smooth'],
            )
            info['wL'] = wL
            info['wR'] = wR
            info['qp_success'] = True

            self.dynamics.set_state(qpos_standing, qvel_zero)
            G_standing = self.dynamics.compute_gravity_vector()
            self.dynamics.set_state(qpos, qvel_zero)
            G_current = self.dynamics.compute_gravity_vector()
            tau_gravity_correction = G_current[6:] - G_standing[6:]
            tau_cg = tau_cg[:self.num_joints] + tau_gravity_correction[:self.num_joints]

        except RuntimeError as e:
            print(f"[Warning] QP failed: {e}")
            tau_cg = np.zeros(self.num_joints)
            info['qp_success'] = False

        q_joints = qpos[7:7 + self.num_joints]
        base_quat = qpos[3:7]

        stance_foot = self._determine_stance_foot(q_joints)

        current_com_est = self._get_com_position(q_joints, stance_foot, base_quat)
        target_com_est = self._get_com_position(
            self.standing_angles, stance_foot, self._standing_yaw_quat
        )

        current_com = current_com_est.copy()
        target_com = target_com_est
        current_com_local = current_com.copy()
        target_com_local = target_com.copy()
        current_com_local[:2] = self._world_xy_to_standing_local(current_com[:2])
        target_com_local[:2] = self._world_xy_to_standing_local(target_com[:2])

        base_drift = self._world_xy_to_standing_local(qpos[:2] - self._base_target_xy)
        base_drift_mag = np.maximum(np.abs(base_drift) - self.base_drift_deadband, 0.0)
        base_drift_corr = np.sign(base_drift) * base_drift_mag * self.base_drift_alpha
        current_com_local[:2] += base_drift_corr

        base_xy_error = base_drift
        base_xy_error_mag = np.maximum(np.abs(base_xy_error) - self.base_xy_integral_deadband, 0.0)
        base_xy_error_eff = np.sign(base_xy_error) * base_xy_error_mag
        self._base_xy_integral += base_xy_error_eff * self.control_dt
        self._base_xy_integral = np.clip(self._base_xy_integral,
                                         -self.base_xy_integral_max,
                                         self.base_xy_integral_max)

        if self._prev_com is None:
            com_vel = np.zeros(3)
        else:
            com_vel_raw = (current_com_local - self._prev_com) / self.control_dt
            if not hasattr(self, '_com_vel_filtered'):
                self._com_vel_filtered = com_vel_raw.copy()
            else:
                a = self.com_vel_filter_alpha
                self._com_vel_filtered = a * com_vel_raw + (1 - a) * self._com_vel_filtered
            com_vel = self._com_vel_filtered
        self._prev_com = current_com_local.copy()

        F_com_local = self._com_pid_control(target_com_local, current_com_local, com_vel)
        F_com_local[:2] += self.base_xy_integral_k * self._base_xy_integral
        base_xy_vel_raw = self._world_xy_to_standing_local(qvel[:2])
        base_xy_vel_mag = np.maximum(np.abs(base_xy_vel_raw) - self.base_xy_damping_deadband, 0.0)
        base_xy_vel_eff = np.sign(base_xy_vel_raw) * base_xy_vel_mag
        F_com_local[:2] -= self.base_xy_damping_k * base_xy_vel_eff
        F_com_local[:2] = np.clip(F_com_local[:2], -self.com_max_force, self.com_max_force)
        F_com = F_com_local.copy()
        F_com[:2] = self._standing_local_xy_to_world(F_com_local[:2])
        self.dynamics.set_state(qpos, qvel)

        foot_L_world = self._foot_world_pos_standing['left']
        foot_R_world = self._foot_world_pos_standing['right']
        pivot_pos = (foot_L_world + foot_R_world) / 2.0
        r_CoM = current_com - pivot_pos
        F_com_local_xy = F_com_local[:2]

        M_pitch = -r_CoM[2] * F_com_local_xy[0]
        M_roll = r_CoM[2] * F_com_local_xy[1]
        tau_com = np.zeros(self.num_joints)
        tau_com[4] = self.ankle_pitch_scale * M_pitch / 2
        tau_com[5] = self.ankle_roll_scale * M_roll / 2
        tau_com[10] = self.ankle_pitch_scale * M_pitch / 2
        tau_com[11] = self.ankle_roll_scale * M_roll / 2

        M_hip = self.hip_roll_scale * r_CoM[2] * F_com_local_xy[1]
        tau_com[2] = M_hip / 2
        tau_com[8] = -M_hip / 2

        M_hip_pitch = -self.hip_pitch_scale * r_CoM[2] * F_com_local_xy[0]
        tau_com[1] = M_hip_pitch / 2
        tau_com[7] = M_hip_pitch / 2

        M_hip_lat = self.hip_roll_lat_scale * r_CoM[2] * F_com_local_xy[1]
        tau_com[1] += M_hip_lat / 2
        tau_com[7] += M_hip_lat / 2

        if self._prev_base_xy is None:
            base_xy_vel = np.zeros(2)
        else:
            base_xy_vel = (qpos[:2] - self._prev_base_xy) / self.control_dt
        self._prev_base_xy = qpos[:2].copy()

        base_xy_error_local = self._world_xy_to_standing_local(qpos[:2] - self._base_target_xy)
        base_xy_vel_local = self._world_xy_to_standing_local(base_xy_vel)
        M_pitch_recenter = (
            self.base_recenter_kp[0] * base_xy_error_local[0]
            + self.base_recenter_kd[0] * base_xy_vel_local[0]
        )
        M_roll_recenter = (
            self.base_recenter_kp[1] * base_xy_error_local[1]
            + self.base_recenter_kd[1] * base_xy_vel_local[1]
        )
        tau_recenter = np.zeros(self.num_joints)
        tau_recenter[4] += 0.5 * self.ankle_pitch_scale * M_pitch_recenter
        tau_recenter[10] += 0.5 * self.ankle_pitch_scale * M_pitch_recenter
        tau_recenter[5] += 0.5 * self.ankle_roll_scale * M_roll_recenter
        tau_recenter[11] += 0.5 * self.ankle_roll_scale * M_roll_recenter
        tau_recenter[1] += 0.5 * self.body_pitch_hip_scale * M_pitch_recenter
        tau_recenter[7] += 0.5 * self.body_pitch_hip_scale * M_pitch_recenter
        tau_recenter[2] += 0.5 * self.body_roll_hip_scale * M_roll_recenter
        tau_recenter[8] -= 0.5 * self.body_roll_hip_scale * M_roll_recenter

        roll, pitch = self._quat_to_roll_pitch_in_standing_frame(qpos[3:7])
        if self._prev_roll is None:
            roll_rate = 0.0
            pitch_rate = 0.0
        else:
            roll_rate = (roll - self._prev_roll) / self.control_dt
            pitch_rate = (pitch - self._prev_pitch) / self.control_dt
        self._prev_roll = roll
        self._prev_pitch = pitch

        pitch_moment = -self.body_pitch_kp * pitch - self.body_pitch_kd * pitch_rate
        roll_moment = -self.body_roll_kp * roll - self.body_roll_kd * roll_rate

        tau_att = np.zeros(self.num_joints)
        tau_att[4] += 0.5 * self.body_pitch_ankle_scale * pitch_moment
        tau_att[10] += 0.5 * self.body_pitch_ankle_scale * pitch_moment
        tau_att[1] += 0.5 * self.body_pitch_hip_scale * pitch_moment
        tau_att[7] += 0.5 * self.body_pitch_hip_scale * pitch_moment

        tau_att[5] += 0.5 * self.body_roll_ankle_scale * roll_moment
        tau_att[11] += 0.5 * self.body_roll_ankle_scale * roll_moment
        tau_att[2] += 0.5 * self.body_roll_hip_scale * roll_moment
        tau_att[8] -= 0.5 * self.body_roll_hip_scale * roll_moment

        info['current_com'] = current_com
        info['target_com'] = target_com
        info['com_vel'] = com_vel
        info['F_com'] = F_com
        info['F_com_local_xy'] = F_com_local_xy
        info['M_pitch'] = M_pitch
        info['M_roll'] = M_roll
        info['base_xy_error'] = self._base_target_xy - qpos[:2]
        info['base_xy_error_local'] = base_xy_error_local
        info['base_drift_corr'] = base_drift_corr
        info['base_xy_integral'] = self._base_xy_integral.copy()
        info['base_xy_vel'] = base_xy_vel
        info['base_xy_vel_local'] = base_xy_vel_local
        info['base_xy_vel_eff'] = base_xy_vel_eff
        info['roll'] = roll
        info['pitch'] = pitch
        info['tau_recenter'] = tau_recenter

        P = self.full_stand_kps_tau * (self.standing_angles - q_joints)
        D = self.full_stand_kds_tau * (np.zeros(self.num_joints) - qvel[6:6 + self.num_joints])
        tau_raw = tau_cg + tau_com + tau_recenter + tau_att + P + D

        info['tau_com'] = tau_com
        info['tau_att'] = tau_att
        info['P'] = P
        info['D'] = D
        info['CG'] = tau_cg

        leg_tau = self._apply_torque_limits(tau_raw)

        arm_tau = self.compute_arm_torque(raw_qpos, raw_qvel)
        tau_full = np.concatenate([leg_tau, arm_tau])
        return tau_full, info

    def compute_arm_torque(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        arm_qpos = qpos[self.arm_qpos_start:self.arm_qpos_start + self.num_arm_joints]
        arm_qvel = qvel[self.arm_qvel_start:self.arm_qvel_start + self.num_arm_joints]
        return self.arm_waist_kps * (self.arm_waist_target - arm_qpos) + self.arm_waist_kds * (0 - arm_qvel)

    def reset(self) -> None:
        self._prev_torque = np.zeros(self.num_joints)
        self._qpos_filtered = None
        self._qvel_filtered = None
        self._prev_com = None
        self._prev_roll = None
        self._prev_pitch = None
        self._prev_base_xy = None
        self._com_error_integral = np.zeros(3)
        self._base_xy_integral = np.zeros(2)
        if hasattr(self, '_com_vel_filtered'):
            del self._com_vel_filtered
        self.dynamics.reset_filters()
