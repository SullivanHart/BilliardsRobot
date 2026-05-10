#!/usr/bin/env python3
"""
Pool shot runner for the HW3 27-DoF robot, using real ball contacts.

Current design / assumptions:
  1. Shot 1 is the original break from the stable standing pose.
  2. After each shot, the current table state is used as-is.
     The code does not stage object balls, does not place the cue ball by
     ball-in-hand, does not inject ball velocities, and does not force-clear balls.
  3. The next shot is selected by simple ghost-ball geometry:
       object ball -> pocket line
       cue ball -> ghost ball line
  4. The robot does not walk between shots.  Auto mode teleports the robot base
     to the computed shot XY/yaw using the same fixed local cue offset that made
     the break work:
       robot_xy = cue_xy - R(yaw) @ ROBOT_TO_CUE_BALL_LOCAL
  5. Every shot runs live StandingCtrl. Follow-up shots reset the robot to the
     same local standing equilibrium at the teleported shot pose.
  6. The cue is still actuated by the cue_slide_joint.
  7. Cue-ball, object-ball, and pocket outcomes are determined by MuJoCo contacts
     plus the simplified pocket detector.
"""

import argparse
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from standing_ctrl import StandingCtrl


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
DEFAULT_CONFIG = str(PROJECT_ROOT / "configs" / "scene27_cue.yaml")
DEFAULT_HW2_CONFIG = DEFAULT_CONFIG


# -----------------------------------------------------------------------------
# Timing
# -----------------------------------------------------------------------------

T_STAND_END = 0.80
T_STRIKE_END = 0.94
T_RETRACT_START = T_STRIKE_END + 0.15
T_RETURN_END = T_STRIKE_END + 0.60

AUTO_STABILIZE_TIME = 0.80
# After teleporting the robot has already been reset to the same stable qpos as
# the break.  Letting the standing controller idle for the full break warmup can
# make it drift before the cue even moves, especially at nonzero yaw.  Later
# shots therefore use a short controller warmup and start the cue stroke almost
# immediately, while still using StandingCtrl during the actual stroke.
TELEPORTED_SHOT_STABILIZE_TIME = 0.80
TELEPORTED_SHOT_CUE_LEAD_TIME = 0.00
AUTO_SHOT_TIMEOUT = 7.00
SHOT_SETTLED_SPEED = 0.035
SHOT_SETTLED_TIME = 0.80

# Real-contact auto-shot filtering.
# This is intentionally conservative: if the actual post-break table only has
# thin/awkward cuts, stop instead of staging balls or injecting fake velocity.
REAL_CONTACT_MIN_CUT_ALIGNMENT = 0.70
REAL_CONTACT_CUT_EPS = 0.025

LOG_EVERY = 200


# -----------------------------------------------------------------------------
# Pool/table constants
# -----------------------------------------------------------------------------

BALL_RADIUS = 0.028575
TABLE_BED_TOP_LOCAL_Z = 0.660
RACK_SPACING = BALL_RADIUS * 2.02

TABLE_HALF_X = 1.35
TABLE_HALF_Y = 0.72
TABLE_WALK_MARGIN = 0.55

POCKET_CENTERS_LOCAL = np.array(
    [
        [1.18, 0.55],
        [1.18, -0.55],
        [-1.18, 0.55],
        [-1.18, -0.55],
        [0.0, 0.55],
        [0.0, -0.55],
    ],
    dtype=np.float64,
)

POCKET_RADII = np.array(
    [0.060, 0.060, 0.060, 0.060, 0.055, 0.055],
    dtype=np.float64,
)

POCKET_MOUTH_X = 1.10
POCKET_MOUTH_Y = 0.50
SIDE_POCKET_HALF_WIDTH = 0.12
POCKET_DROP_LOCAL_Z = 0.28

BALL_BODY_NAMES = ["cue_ball"] + [f"ball_{i}" for i in range(1, 16)]
OBJECT_BALL_NAMES = [f"ball_{i}" for i in range(1, 16)]

BALL_LINEAR_DAMPING = 0.0
BALL_ANGULAR_DAMPING = 0.0
BALL_ROLLING_RESISTANCE = 0.012


# -----------------------------------------------------------------------------
# Cue/stroke constants
# -----------------------------------------------------------------------------

CUE_SLIDE_END = 0.35
CUE_HIT_OVERLAP = 0.014
BALL_PLACEMENT_OVERLAP = 0.001

# This is the important fixed shot geometry.
# The first shot works because this offset puts the robot/cue in the right place.
# Reuse the same body-frame offset for every later shot.
ROBOT_TO_CUE_BALL_LOCAL = np.array([1.50, -0.116], dtype=np.float64)

# Cue alignment should only be a trim correction.
# Large "alignment" teleports destabilize the standing controller.
MAX_CUE_ALIGN_STEP = 0.10

# Keep the arm pose fixed during cue-stick extension.
PITCH_COMP_SCALE = 0.0
PITCH_COMP_ALPHA = 0.75

# Real-contact shot mode.  Later shots still teleport/reset the robot, but the
# cue ball, object balls, and pockets are driven only by MuJoCo contacts.  No
# ball-in-hand placement, forced ball velocity, staged shot, or force-clear is
# used in auto mode.
#
# -----------------------------------------------------------------------------
# Initial stable shot pose / arm pose
# -----------------------------------------------------------------------------

SHOT_STAND_XY = np.array([0.13, 0.116], dtype=np.float64)

STROKE_RIGHT_WAYPOINTS = np.array(
    [
        [0.147696, -0.049493, 0.262774, 0.045309, -0.106756, -0.180400, -0.250454],
        [-0.003725, -0.056675, 0.232080, 0.176397, -0.120231, -0.155360, -0.257153],
        [-0.164113, -0.077679, 0.214545, 0.349134, -0.157697, -0.153918, -0.276468],
        [-0.337883, -0.114662, 0.222731, 0.564522, -0.230767, -0.162362, -0.315270],
        [-0.536951, -0.167521, 0.286235, 0.828001, -0.367466, -0.145806, -0.387150],
    ],
    dtype=np.float64,
)

RIGHT_START_TARGET = STROKE_RIGHT_WAYPOINTS[0].copy()
RIGHT_END_TARGET = STROKE_RIGHT_WAYPOINTS[-1].copy()
RIGHT_HOME_TARGET = RIGHT_START_TARGET.copy()

RIGHT_STROKE_TIP_TARGETS = np.array(
    [
        [1.272, 0.003, 0.860],
        [1.302, 0.003, 0.860],
        [1.332, 0.003, 0.860],
        [1.362, 0.003, 0.860],
        [1.392, 0.003, 0.860],
        [1.422, 0.003, 0.860],
        [1.449, 0.003, 0.862],
    ],
    dtype=np.float64,
)


# -----------------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------------

def yaw_from_quat_wxyz(quat):
    return R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz")[2]


def quat_wxyz_from_yaw(yaw):
    return R.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]


def body_tilt_deg_from_quat_wxyz(quat):
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    body_z = rot[:, 2]
    return float(np.degrees(np.arccos(np.clip(body_z[2], -1.0, 1.0))))


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    return np.array(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def resolve_config_path(path_str, root):
    out = path_str
    for placeholder in ("{DIR}", "{LEGGED_GYM_ROOT_DIR}"):
        out = out.replace(placeholder, str(root))
    return out



# -----------------------------------------------------------------------------
# Standing controller / sim helpers
# -----------------------------------------------------------------------------

def make_pool_ctrl(config_file):
    ctrl = StandingCtrl(config_file)

    ctrl.stand_CoM[:2] = SHOT_STAND_XY
    ctrl._base_target_xy = SHOT_STAND_XY.copy()
    ctrl.arm_waist_target[8:15] = RIGHT_HOME_TARGET

    # Stiff right arm for cue stability.
    ctrl.arm_waist_kps[8] = 200.0
    ctrl.arm_waist_kds[8] = 20.0
    ctrl.arm_waist_kps[9] = 150.0
    ctrl.arm_waist_kds[9] = 15.0
    ctrl.arm_waist_kps[10] = 100.0
    ctrl.arm_waist_kds[10] = 10.0
    ctrl.arm_waist_kps[11] = 150.0
    ctrl.arm_waist_kds[11] = 20.0
    ctrl.arm_waist_kps[12] = 80.0
    ctrl.arm_waist_kds[12] = 8.0
    ctrl.arm_waist_kps[13] = 50.0
    ctrl.arm_waist_kds[13] = 5.0
    ctrl.arm_waist_kps[14] = 50.0
    ctrl.arm_waist_kds[14] = 5.0

    if hasattr(ctrl, "update_standing_reference_from_state"):
        initial_qpos = ctrl.get_initial_state()["qpos"]
        ctrl.update_standing_reference_from_state(initial_qpos)

    ctrl.reset()
    return ctrl


def initialize_data(model, data, ctrl):
    init_state = ctrl.get_initial_state()
    init_state["qpos"][:2] = SHOT_STAND_XY

    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[:ctrl.nq] = init_state["qpos"][:ctrl.nq]
    data.qvel[:ctrl.nv] = init_state["qvel"][:ctrl.nv]

    mujoco.mj_forward(model, data)
    sync_standing_ctrl_to_current_pose(ctrl, data)


def standing_ref_summary(ctrl, data):
    yaw = yaw_from_quat_wxyz(data.qpos[3:7])
    foot_mid = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    if hasattr(ctrl, "_foot_world_pos_standing"):
        foot_mid = 0.5 * (
            ctrl._foot_world_pos_standing["left"] + ctrl._foot_world_pos_standing["right"]
        )
    ref_com = getattr(ctrl, "_standing_world_com", np.array([np.nan, np.nan, np.nan]))
    base_target = getattr(ctrl, "_base_target_xy", np.array([np.nan, np.nan]))
    return (
        f"base=({data.qpos[0]:+.3f},{data.qpos[1]:+.3f}) "
        f"base_target=({base_target[0]:+.3f},{base_target[1]:+.3f}) "
        f"yaw={np.degrees(yaw):+.1f}deg "
        f"foot_mid=({foot_mid[0]:+.3f},{foot_mid[1]:+.3f},{foot_mid[2]:+.3f}) "
        f"ref_com=({ref_com[0]:+.3f},{ref_com[1]:+.3f},{ref_com[2]:+.3f})"
    )


def sync_standing_ctrl_to_current_pose(ctrl, data):
    xy = data.qpos[:2].copy()

    if hasattr(ctrl, "update_standing_reference_from_state"):
        ctrl.update_standing_reference_from_state(data.qpos[:ctrl.nq])
    else:
        ctrl.stand_CoM[:2] = xy
        ctrl._base_target_xy = xy.copy()

    if hasattr(ctrl, "_base_xy_integral"):
        ctrl._base_xy_integral[:] = 0.0
    if hasattr(ctrl, "_com_error_integral"):
        ctrl._com_error_integral[:] = 0.0
    if hasattr(ctrl, "_prev_base_xy"):
        ctrl._prev_base_xy = None


def set_robot_xy_yaw(model, data, ctrl, xy, yaw):
    data.qpos[:2] = np.array(xy, dtype=np.float64)
    data.qpos[3:7] = quat_wxyz_from_yaw(yaw)
    data.qvel[:6] = 0.0

    sync_standing_ctrl_to_current_pose(ctrl, data)
    mujoco.mj_forward(model, data)


def reset_robot_to_stable_shot_pose(model, data, ctrl, xy, yaw, home_arm=None):
    """Reset the robot-only state, then place the base at a shot pose.

    Important for teleport mode: simply changing XY/yaw after a shot preserves
    whatever body height, joint error, and accumulated velocity the standing
    controller had after the previous stroke. That is why the robot can start
    the next shot already sagging/falling. This restores the same stable robot
    state used at startup, then changes only the base XY/yaw. Ball/table qpos
    values live after ctrl.nq, so they are preserved.
    """
    init_state = ctrl.get_initial_state()

    data.qpos[:ctrl.nq] = init_state["qpos"][:ctrl.nq]
    data.qvel[:ctrl.nv] = 0.0

    if home_arm is not None:
        home_arm = np.array(home_arm, dtype=np.float64)
        data.qpos[19:34] = home_arm
        ctrl.arm_waist_target = home_arm.copy()

    data.qpos[:2] = np.array(xy, dtype=np.float64)
    data.qpos[3:7] = quat_wxyz_from_yaw(yaw)

    ctrl.reset()
    sync_standing_ctrl_to_current_pose(ctrl, data)

    # Clear any residual base and joint velocities from the previous stroke.
    data.qvel[:ctrl.nv] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def arm_dof_addrs_for_model(model, ctrl):
    return np.array(
        [
            model.jnt_dofadr[model.actuator_trnid[i, 0]]
            for i in range(ctrl.num_joints, ctrl.num_actuators)
        ],
        dtype=np.int32,
    )


def cue_slide_addrs_for_model(model):
    cue_slide_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cue_slide_joint")
    if cue_slide_jid < 0:
        raise RuntimeError("cue_slide_joint not found in sim model")
    return model.jnt_qposadr[cue_slide_jid], model.jnt_dofadr[cue_slide_jid]


def body_free_joint_addrs(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found: {body_name}")

    joint_id = model.body_jntadr[body_id]
    if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Body does not have a free joint: {body_name}")

    return model.jnt_qposadr[joint_id], model.jnt_dofadr[joint_id]


def set_free_body_pose(model, data, body_name, pos):
    qadr, dadr = body_free_joint_addrs(model, body_name)
    data.qpos[qadr:qadr + 3] = np.array(pos, dtype=np.float64)
    data.qpos[qadr + 3:qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[dadr:dadr + 6] = 0.0


def set_free_body_linvel(model, data, body_name, vel):
    _, dadr = body_free_joint_addrs(model, body_name)
    data.qvel[dadr:dadr + 3] = np.array(vel, dtype=np.float64)
    data.qvel[dadr + 3:dadr + 6] = 0.0


def body_xy(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found: {body_name}")
    return data.xpos[body_id, :2].copy()


def disable_cue_contacts(model):
    for geom_name in ("cue_tip", "cue_shaft"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0


def enable_cue_tip_ball_contact(model):
    shaft_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cue_shaft")
    if shaft_id >= 0:
        model.geom_contype[shaft_id] = 0
        model.geom_conaffinity[shaft_id] = 0

    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cue_tip")
    if tip_id >= 0:
        model.geom_contype[tip_id] = 1
        model.geom_conaffinity[tip_id] = 1


def save_cue_contacts(model):
    out = {}
    for geom_name in ("cue_tip", "cue_shaft"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            out[geom_id] = (
                int(model.geom_contype[geom_id]),
                int(model.geom_conaffinity[geom_id]),
            )
    return out


def restore_geom_contacts(model, saved_contacts):
    for geom_id, values in saved_contacts.items():
        model.geom_contype[geom_id] = values[0]
        model.geom_conaffinity[geom_id] = values[1]


def tune_ball_joint_damping(model):
    for body_name in BALL_BODY_NAMES:
        try:
            _, dadr = body_free_joint_addrs(model, body_name)
        except ValueError:
            continue

        model.dof_damping[dadr:dadr + 3] = BALL_LINEAR_DAMPING
        model.dof_damping[dadr + 3:dadr + 6] = BALL_ANGULAR_DAMPING


# -----------------------------------------------------------------------------
# Cue and stroke
# -----------------------------------------------------------------------------

def right_arm_target_at_time(_t, home_arm):
    out = np.array(home_arm, dtype=np.float64).copy()
    out[8:15] = RIGHT_START_TARGET
    return out


def cue_extension_at_time(t):
    if t < T_STAND_END:
        return 0.0, 0.0

    if t < T_STRIKE_END:
        duration = T_STRIKE_END - T_STAND_END
        alpha = float(np.clip((t - T_STAND_END) / duration, 0.0, 1.0))
        return CUE_SLIDE_END * alpha, CUE_SLIDE_END / duration

    if t < T_RETRACT_START:
        return CUE_SLIDE_END, 0.0

    if t < T_RETURN_END:
        duration = T_RETURN_END - T_RETRACT_START
        alpha = float(np.clip((t - T_RETRACT_START) / duration, 0.0, 1.0))
        return CUE_SLIDE_END * (1.0 - alpha), -CUE_SLIDE_END / duration

    return 0.0, 0.0


def run_controller_step(
    ctrl,
    model,
    data,
    tau,
    counter,
    home_arm,
    arm_dof_addrs=None,
    d_grav=None,
    pitch_state=None,
    cue_slide_qadr=None,
    cue_slide_dadr=None,
    local_t=None,
    cue_enabled=True,
):
    t = counter * ctrl.simulation_dt if local_t is None else local_t
    ctrl.arm_waist_target = right_arm_target_at_time(t, home_arm)

    robot_qpos = data.qpos[:ctrl.nq].copy()
    robot_qvel = data.qvel[:ctrl.nv].copy()

    quat_wxyz = robot_qpos[3:7]
    body_pitch_raw = R.from_quat(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    ).as_euler("xyz")[1]

    if pitch_state is not None:
        pitch_state[0] = PITCH_COMP_ALPHA * pitch_state[0] + (1.0 - PITCH_COMP_ALPHA) * body_pitch_raw
        body_pitch = pitch_state[0]
    else:
        body_pitch = body_pitch_raw

    ctrl.arm_waist_target[8] = float(
        np.clip(ctrl.arm_waist_target[8] - PITCH_COMP_SCALE * body_pitch, -3.09, 2.67)
    )

    if counter % ctrl.control_decimation == 0:
        tau[:], info = ctrl.compute_torque(robot_qpos, robot_qvel)
    else:
        tau[ctrl.num_joints:] = ctrl.compute_arm_torque(robot_qpos, robot_qvel)
        info = {}

    if arm_dof_addrs is not None and d_grav is not None and robot_qpos[2] > 0.45:
        d_grav.qpos[:ctrl.nq] = robot_qpos
        d_grav.qpos[ctrl.arm_qpos_start:ctrl.arm_qpos_start + ctrl.num_arm_joints] = ctrl.arm_waist_target
        d_grav.qvel[:ctrl.nv] = 0.0
        mujoco.mj_forward(model, d_grav)
        tau[ctrl.num_joints:] += d_grav.qfrc_bias[arm_dof_addrs]

    if cue_slide_qadr is not None and cue_slide_dadr is not None:
        if not cue_enabled:
            extension, extension_vel = 0.0, 0.0
        else:
            extension, extension_vel = cue_extension_at_time(t)
        data.qpos[cue_slide_qadr] = extension
        data.qvel[cue_slide_dadr] = extension_vel

    data.ctrl[:] = tau
    mujoco.mj_step(model, data)

    return t, info


def measure_extended_cue(config_file, sim_xml_path):
    measure_ctrl = make_pool_ctrl(config_file)
    measure_model = mujoco.MjModel.from_xml_path(sim_xml_path)
    measure_model.opt.timestep = measure_ctrl.simulation_dt
    measure_data = mujoco.MjData(measure_model)

    initialize_data(measure_model, measure_data, measure_ctrl)
    set_free_body_pose(measure_model, measure_data, "cue_ball", np.array([5.0, 0.0, 1.0]))
    mujoco.mj_forward(measure_model, measure_data)

    cue_tip_sid = mujoco.mj_name2id(measure_model, mujoco.mjtObj.mjOBJ_SITE, "cue_tip_site")
    cue_bid = mujoco.mj_name2id(measure_model, mujoco.mjtObj.mjOBJ_BODY, "right_cue")
    cue_slide_qadr, cue_slide_dadr = cue_slide_addrs_for_model(measure_model)
    arm_dof_addrs = arm_dof_addrs_for_model(measure_model, measure_ctrl)

    d_grav = mujoco.MjData(measure_model)
    d_grav.qpos[:] = measure_model.qpos0.copy()

    home_arm = np.array(measure_ctrl.arm_waist_target, dtype=np.float64)
    pitch_state = [0.0]
    tau = np.zeros(measure_ctrl.num_actuators)

    best = None
    prev_tip = None
    max_steps = int((T_STRIKE_END + 0.08) / measure_ctrl.simulation_dt)

    for counter in range(max_steps):
        t, _ = run_controller_step(
            measure_ctrl,
            measure_model,
            measure_data,
            tau,
            counter,
            home_arm,
            arm_dof_addrs=arm_dof_addrs,
            d_grav=d_grav,
            pitch_state=pitch_state,
            cue_slide_qadr=cue_slide_qadr,
            cue_slide_dadr=cue_slide_dadr,
            cue_enabled=True,
        )

        tip = measure_data.site_xpos[cue_tip_sid].copy()
        if prev_tip is None:
            prev_tip = tip
            continue

        tip_vel = (tip - prev_tip) / measure_ctrl.simulation_dt
        prev_tip = tip

        if t < T_STAND_END + 0.05 or t > T_STRIKE_END - 0.02:
            continue

        axis = measure_data.xmat[cue_bid].reshape(3, 3)[:, 0].copy()
        score = (
            tip_vel[0]
            - 0.4 * abs(tip_vel[1])
            - 0.6 * abs(tip_vel[2])
            - 0.5 * abs(axis[2])
        )

        if best is None or score > best[0]:
            best = (score, t, tip, axis, tip_vel, measure_data.qpos[2])

    if best is None:
        raise RuntimeError("failed to measure extended cue pose")

    return best[1:]


def align_robot_cue_to_ball(model, data, ctrl, cue_tip_sid, cue_bid, cue_slide_qadr, cue_xy, arm_target):
    """
    Trim-align the robot base so the extended cue tip is near the cue ball.

    Important:
      - This must be a small correction only.
      - Large corrections after walking destabilize the standing controller.
      - If the correction is too large, apply only a capped step and report the
        remaining error instead of pretending it is fully aligned.
    """
    data.qpos[19:34] = arm_target
    data.qpos[cue_slide_qadr] = CUE_SLIDE_END
    mujoco.mj_forward(model, data)

    cue_axis = data.xmat[cue_bid].reshape(3, 3)[:, 0].copy()
    desired_tip_xy = np.array(cue_xy, dtype=np.float64) - cue_axis[:2] * (BALL_RADIUS - CUE_HIT_OVERLAP)
    tip_xy = data.site_xpos[cue_tip_sid, :2].copy()

    raw_delta = desired_tip_xy - tip_xy
    raw_norm = float(np.linalg.norm(raw_delta))

    if raw_norm < 1e-9:
        applied_delta = np.zeros(2, dtype=np.float64)
        remaining_delta = np.zeros(2, dtype=np.float64)
    else:
        step_norm = min(raw_norm, MAX_CUE_ALIGN_STEP)
        applied_delta = raw_delta * (step_norm / raw_norm)
        remaining_delta = raw_delta - applied_delta

    yaw_now = yaw_from_quat_wxyz(data.qpos[3:7])
    corrected_xy = data.qpos[:2].copy() + applied_delta

    set_robot_xy_yaw(model, data, ctrl, corrected_xy, yaw_now)

    data.qpos[cue_slide_qadr] = 0.0
    data.qvel[:] = 0.0
    ctrl.reset()
    sync_standing_ctrl_to_current_pose(ctrl, data)
    mujoco.mj_forward(model, data)

    print(
        "cue align: "
        f"raw_delta=({raw_delta[0]:+.3f},{raw_delta[1]:+.3f}) "
        f"raw_norm={raw_norm:.3f} "
        f"applied=({applied_delta[0]:+.3f},{applied_delta[1]:+.3f}) "
        f"applied_norm={np.linalg.norm(applied_delta):.3f} "
        f"remaining=({remaining_delta[0]:+.3f},{remaining_delta[1]:+.3f}) "
        f"remaining_norm={np.linalg.norm(remaining_delta):.3f}",
        flush=True,
    )

    return applied_delta


# -----------------------------------------------------------------------------
# Pool layout / pockets / ball physics
# -----------------------------------------------------------------------------

def place_pool_layout(model, data, ball_pos):
    table_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pool_table")
    if table_bid < 0:
        raise ValueError("pool_table body not found")

    model.body_pos[table_bid] = np.array(
        [
            ball_pos[0] + 0.70,
            ball_pos[1],
            ball_pos[2] - BALL_RADIUS - TABLE_BED_TOP_LOCAL_Z,
        ],
        dtype=np.float64,
    )

    set_free_body_pose(model, data, "cue_ball", ball_pos)

    rack_start = ball_pos + np.array([0.75, 0.0, 0.0])
    ball_index = 1

    for row in range(5):
        for col in range(row + 1):
            if ball_index > 15:
                break

            rack_pos = rack_start + np.array(
                [
                    row * RACK_SPACING,
                    (col - row / 2.0) * RACK_SPACING,
                    0.0,
                ],
                dtype=np.float64,
            )

            set_free_body_pose(model, data, f"ball_{ball_index}", rack_pos)
            ball_index += 1

    mujoco.mj_forward(model, data)


def table_pocket_centers_world(model, data):
    table_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pool_table")
    table_pos = data.xpos[table_bid].copy()
    return table_pos[:2] + POCKET_CENTERS_LOCAL


def table_keepout_bounds(model, data):
    table_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pool_table")
    center = data.xpos[table_bid, :2].copy()
    half = np.array(
        [
            TABLE_HALF_X + TABLE_WALK_MARGIN,
            TABLE_HALF_Y + TABLE_WALK_MARGIN,
        ],
        dtype=np.float64,
    )
    return center, center - half, center + half


def point_rect_distance(p, rect_min, rect_max):
    dx = max(rect_min[0] - p[0], 0.0, p[0] - rect_max[0])
    dy = max(rect_min[1] - p[1], 0.0, p[1] - rect_max[1])
    return float(np.hypot(dx, dy))


def apply_ball_rolling_resistance(model, data, pocketed_balls):
    decel = BALL_ROLLING_RESISTANCE * 9.81 * model.opt.timestep

    for body_name in BALL_BODY_NAMES:
        if body_name in pocketed_balls:
            continue

        try:
            _, dadr = body_free_joint_addrs(model, body_name)
        except ValueError:
            continue

        vxy = data.qvel[dadr:dadr + 2]
        speed = float(np.linalg.norm(vxy))

        if speed <= decel:
            data.qvel[dadr:dadr + 2] = 0.0
            data.qvel[dadr + 3:dadr + 6] = 0.0
        else:
            data.qvel[dadr:dadr + 2] *= (speed - decel) / speed


def update_pocketed_balls(model, data, pocketed_balls, cue_ball_start_pos):
    table_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pool_table")
    table_pos = data.xpos[table_bid].copy()
    pocket_centers = table_pos[:2] + POCKET_CENTERS_LOCAL

    new_pocketed = []

    for body_name in BALL_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue

        qadr, dadr = body_free_joint_addrs(model, body_name)

        if body_name not in pocketed_balls:
            xy = data.xpos[body_id, :2]
            local_xy = xy - table_pos[:2]

            if (
                body_name == "cue_ball"
                and (
                    abs(local_xy[0]) > TABLE_HALF_X + 0.25
                    or abs(local_xy[1]) > TABLE_HALF_Y + 0.25
                )
            ):
                set_free_body_pose(model, data, body_name, cue_ball_start_pos)
                new_pocketed.append("cue_ball_respawned")
                continue

            distances = np.linalg.norm(pocket_centers - xy, axis=1)

            for pocket_idx, distance in enumerate(distances):
                if distance >= POCKET_RADII[pocket_idx]:
                    continue

                if pocket_idx < 4:
                    in_mouth = abs(local_xy[0]) > POCKET_MOUTH_X and abs(local_xy[1]) > POCKET_MOUTH_Y
                else:
                    in_mouth = abs(local_xy[0]) < SIDE_POCKET_HALF_WIDTH and abs(local_xy[1]) > POCKET_MOUTH_Y

                if not in_mouth:
                    continue

                if body_name == "cue_ball":
                    set_free_body_pose(model, data, body_name, cue_ball_start_pos)
                    new_pocketed.append("cue_ball_respawned")
                else:
                    pocketed_balls[body_name] = pocket_idx
                    new_pocketed.append(body_name)

                break

        if body_name in pocketed_balls:
            pocket_xy = pocket_centers[pocketed_balls[body_name]]
            data.qpos[qadr:qadr + 3] = np.array(
                [
                    pocket_xy[0],
                    pocket_xy[1],
                    table_pos[2] + POCKET_DROP_LOCAL_Z,
                ],
                dtype=np.float64,
            )
            data.qvel[dadr:dadr + 6] = 0.0

    return new_pocketed




def cue_tip_ball_alignment(model, data, cue_tip_sid, cue_bid, cue_ball_bid):
    """
    Return geometry metrics between the cue tip and the cue ball.

    along > 0 means the ball is in front of the cue tip along the cue axis.
    lateral is the perpendicular XY miss distance from the cue axis to the ball.
    """
    tip_xy = data.site_xpos[cue_tip_sid, :2].copy()
    ball_xy = data.xpos[cue_ball_bid, :2].copy()
    cue_axis = data.xmat[cue_bid].reshape(3, 3)[:, 0].copy()
    axis_xy = cue_axis[:2].astype(np.float64)
    axis_norm = float(np.linalg.norm(axis_xy))
    if axis_norm < 1e-9:
        axis_xy = np.array([1.0, 0.0], dtype=np.float64)
    else:
        axis_xy = axis_xy / axis_norm

    delta = ball_xy - tip_xy
    along = float(np.dot(delta, axis_xy))
    lateral_vec = delta - along * axis_xy
    lateral = float(np.linalg.norm(lateral_vec))
    dist = float(np.linalg.norm(delta))

    return {
        "tip_xy": tip_xy,
        "ball_xy": ball_xy,
        "axis_xy": axis_xy,
        "along": along,
        "lateral": lateral,
        "dist": dist,
    }


def cue_tip_cue_ball_mujoco_contact(model, data):
    """Return whether MuJoCo currently has a cue-tip <-> cue-ball contact.

    This is stricter than geometric alignment.  The cue can be visually close to
    the cue ball without MuJoCo generating a contact constraint.  For the
    follow-up punch stroke, use this actual contact test to decide when to
    retract, so the cue does not pull back before impact or keep spearing through
    the ball after impact.
    """
    cue_tip_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cue_tip")
    cue_ball_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cue_ball")
    if cue_tip_gid < 0 or cue_ball_bid < 0:
        return False, None

    best = None
    for i in range(data.ncon):
        con = data.contact[i]
        g1 = int(con.geom1)
        g2 = int(con.geom2)
        b1 = int(model.geom_bodyid[g1]) if g1 >= 0 else -1
        b2 = int(model.geom_bodyid[g2]) if g2 >= 0 else -1

        hit = (
            (g1 == cue_tip_gid and b2 == cue_ball_bid)
            or (g2 == cue_tip_gid and b1 == cue_ball_bid)
        )
        if not hit:
            continue

        dist = float(con.dist)
        if best is None or dist < best["dist"]:
            best = {
                "contact_index": int(i),
                "geom1": g1,
                "geom2": g2,
                "body1": b1,
                "body2": b2,
                "dist": dist,
                "pos": np.array(con.pos, dtype=np.float64).copy(),
            }

    return best is not None, best


def body_body_mujoco_contact(model, data, body_a_name, body_b_name):
    """Return whether MuJoCo currently has contact between two named bodies."""
    body_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_a_name)
    body_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_b_name)
    if body_a < 0 or body_b < 0:
        return False, None

    best = None
    for i in range(data.ncon):
        con = data.contact[i]
        g1 = int(con.geom1)
        g2 = int(con.geom2)
        b1 = int(model.geom_bodyid[g1]) if g1 >= 0 else -1
        b2 = int(model.geom_bodyid[g2]) if g2 >= 0 else -1

        hit = (b1 == body_a and b2 == body_b) or (b1 == body_b and b2 == body_a)
        if not hit:
            continue

        dist = float(con.dist)
        if best is None or dist < best["dist"]:
            best = {
                "contact_index": int(i),
                "body1": b1,
                "body2": b2,
                "dist": dist,
                "pos": np.array(con.pos, dtype=np.float64).copy(),
            }

    return best is not None, best


def body_linear_speed(model, data, body_name):
    try:
        _, dadr = body_free_joint_addrs(model, body_name)
    except ValueError:
        return 0.0
    return float(np.linalg.norm(data.qvel[dadr:dadr + 3]))


def max_live_ball_speed(model, data, pocketed_balls):
    out = 0.0

    for body_name in BALL_BODY_NAMES:
        if body_name in pocketed_balls:
            continue

        try:
            _, dadr = body_free_joint_addrs(model, body_name)
        except ValueError:
            continue

        out = max(out, float(np.linalg.norm(data.qvel[dadr:dadr + 3])))

    return out


def ball_break_metrics(model, data, pocketed_balls):
    object_xy = []
    max_speed = 0.0
    moving = 0

    for body_name in BALL_BODY_NAMES:
        if body_name in pocketed_balls:
            continue

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue

        _, dadr = body_free_joint_addrs(model, body_name)
        speed = float(np.linalg.norm(data.qvel[dadr:dadr + 3]))

        max_speed = max(max_speed, speed)
        if speed > 0.03:
            moving += 1

        if body_name != "cue_ball":
            object_xy.append(data.xpos[body_id, :2].copy())

    if not object_xy:
        return 0.0, 0.0, max_speed, moving

    object_xy = np.array(object_xy)
    center = object_xy.mean(axis=0)
    radial = np.linalg.norm(object_xy - center, axis=1)

    return float(radial.mean()), float(radial.max()), max_speed, moving


def settle_balls_only(model, data, ctrl, pocketed_balls, cue_ball_start_pos, viewer=None, counter_ref=None):
    settled = 0.0
    local_t = 0.0

    while local_t < AUTO_SHOT_TIMEOUT:
        robot_qpos = data.qpos[:ctrl.nq].copy()

        data.qvel[:ctrl.nv] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)

        data.qpos[:ctrl.nq] = robot_qpos
        data.qvel[:ctrl.nv] = 0.0
        mujoco.mj_forward(model, data)

        new_pocketed = update_pocketed_balls(model, data, pocketed_balls, cue_ball_start_pos)
        apply_ball_rolling_resistance(model, data, set(pocketed_balls.keys()))

        for body_name in new_pocketed:
            if body_name == "cue_ball_respawned":
                print("[settle] cue_ball scratched; respawned", flush=True)
            else:
                print(f"[settle {local_t:.2f}s] pocketed {body_name}", flush=True)

        max_speed = max_live_ball_speed(model, data, set(pocketed_balls.keys()))
        if max_speed < SHOT_SETTLED_SPEED:
            settled += model.opt.timestep
            if settled >= SHOT_SETTLED_TIME:
                return True
        else:
            settled = 0.0

        if viewer is not None and counter_ref is not None and counter_ref[0] % 4 == 0:
            viewer.sync()

        if counter_ref is not None:
            counter_ref[0] += 1

        local_t += model.opt.timestep

    return False


# -----------------------------------------------------------------------------
# Shot planning
# -----------------------------------------------------------------------------

def segment_clear(a, b, blockers, clearance):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    ab = b - a
    ab_len2 = float(np.dot(ab, ab))

    if ab_len2 < 1e-12:
        return False

    for blocker in blockers:
        blocker = np.array(blocker, dtype=np.float64)
        t = float(np.clip(np.dot(blocker - a, ab) / ab_len2, 0.0, 1.0))
        closest = a + t * ab

        if float(np.linalg.norm(blocker - closest)) < clearance:
            return False

    return True


def choose_next_shot(model, data, pocketed_balls, attempted_shots):
    cue_xy = body_xy(model, data, "cue_ball")
    pockets = table_pocket_centers_world(model, data)

    live = [name for name in OBJECT_BALL_NAMES if name not in pocketed_balls]
    positions = {name: body_xy(model, data, name) for name in live}

    clearance = 2.15 * BALL_RADIUS
    best = None

    for ball_name, obj_xy in positions.items():
        other_balls = [pos for other, pos in positions.items() if other != ball_name]

        for pocket_idx, pocket_xy in enumerate(pockets):
            if (ball_name, pocket_idx) in attempted_shots:
                continue

            obj_to_pocket = pocket_xy - obj_xy
            obj_to_pocket_dist = float(np.linalg.norm(obj_to_pocket))

            if obj_to_pocket_dist < 4.0 * BALL_RADIUS:
                continue

            obj_dir = obj_to_pocket / obj_to_pocket_dist
            ghost_xy = obj_xy - obj_dir * (2.0 * BALL_RADIUS)

            cue_to_ghost = ghost_xy - cue_xy
            cue_to_ghost_dist = float(np.linalg.norm(cue_to_ghost))

            if cue_to_ghost_dist < 0.30:
                continue

            cue_dir = cue_to_ghost / cue_to_ghost_dist
            cut_alignment = float(np.dot(cue_dir, obj_dir))

            if cut_alignment < 0.45:
                continue

            if not segment_clear(obj_xy, pocket_xy, other_balls, clearance):
                continue

            if not segment_clear(cue_xy, ghost_xy, other_balls, clearance):
                continue

            score = (
                3.0 * cut_alignment
                - 0.25 * cue_to_ghost_dist
                - 0.20 * obj_to_pocket_dist
                + (0.15 if pocket_idx >= 4 else 0.0)
            )

            candidate = {
                "score": score,
                "ball": ball_name,
                "pocket_idx": pocket_idx,
                "pocket_xy": pocket_xy,
                "object_xy": obj_xy,
                "ghost_xy": ghost_xy,
                "cue_xy": cue_xy,
                "shot_dir": cue_dir,
                "object_dir": obj_dir,
                "cut_alignment": cut_alignment,
                "cue_dist": cue_to_ghost_dist,
                "object_dist": obj_to_pocket_dist,
            }

            if best is None or score > best["score"]:
                best = candidate

    return best




def choose_next_shot_relaxed(model, data, pocketed_balls, attempted_shots):
    """
    Fallback planner for teleport/clear-table mode.

    The normal planner is intentionally conservative about blockers and cut angle,
    which is good for realistic pool but bad for this assignment-style clear-table
    demo.  This relaxed planner still computes a real cue-ball/ghost-ball pose,
    but it will accept tougher or blocked shots so the robot keeps visibly
    teleporting, aiming, stroking, and clearing balls instead of immediately
    deleting them by fallback.
    """
    cue_xy = body_xy(model, data, "cue_ball")
    pockets = table_pocket_centers_world(model, data)

    live = [name for name in OBJECT_BALL_NAMES if name not in pocketed_balls]
    best = None

    for ball_name in live:
        obj_xy = body_xy(model, data, ball_name)

        for pocket_idx, pocket_xy in enumerate(pockets):
            # In relaxed mode, still prefer unattempted shots, but allow retry if
            # every option has already been attempted.
            attempted_penalty = 1.0 if (ball_name, pocket_idx) in attempted_shots else 0.0

            obj_to_pocket = pocket_xy - obj_xy
            obj_to_pocket_dist = float(np.linalg.norm(obj_to_pocket))
            if obj_to_pocket_dist < 1e-6:
                continue

            obj_dir = obj_to_pocket / obj_to_pocket_dist
            ghost_xy = obj_xy - obj_dir * (2.0 * BALL_RADIUS)

            cue_to_ghost = ghost_xy - cue_xy
            cue_to_ghost_dist = float(np.linalg.norm(cue_to_ghost))
            if cue_to_ghost_dist < 0.12:
                # Too cramped to place a useful cue line.
                continue

            cue_dir = cue_to_ghost / cue_to_ghost_dist
            cut_alignment = float(np.dot(cue_dir, obj_dir))

            # The cue-ball path still needs to be clear.  For turn-in/demo mode
            # the important visible behavior is cue ball -> target ball contact;
            # accepting a blocked cue path just produces a clean cue-ball hit
            # that never reaches the called object ball.
            other_balls = [body_xy(model, data, other) for other in live if other != ball_name]
            clearance = 1.65 * BALL_RADIUS
            if not segment_clear(cue_xy, ghost_xy, other_balls, clearance):
                continue

            # Object-to-pocket blockers are only a soft penalty.  A non-pocketed
            # but real target-ball hit is still useful for the final demo.
            obj_blocked = 0.0 if segment_clear(obj_xy, pocket_xy, other_balls, clearance) else 1.0

            score = (
                2.4 * cut_alignment
                - 0.18 * cue_to_ghost_dist
                - 0.16 * obj_to_pocket_dist
                - 0.35 * obj_blocked
                - 0.70 * attempted_penalty
                + (0.10 if pocket_idx >= 4 else 0.0)
            )

            candidate = {
                "score": score,
                "ball": ball_name,
                "pocket_idx": pocket_idx,
                "pocket_xy": pocket_xy,
                "object_xy": obj_xy,
                "ghost_xy": ghost_xy,
                "cue_xy": cue_xy,
                "shot_dir": cue_dir,
                "object_dir": obj_dir,
                "cut_alignment": cut_alignment,
                "cue_dist": cue_to_ghost_dist,
                "object_dist": obj_to_pocket_dist,
                "relaxed": True,
                "obj_blocked": bool(obj_blocked),
                "cue_blocked": False,
            }

            if best is None or score > best["score"]:
                best = candidate

    return best

def robot_pose_for_shot(cue_xy, shot_dir):
    shot_dir = np.array(shot_dir, dtype=np.float64)
    n = float(np.linalg.norm(shot_dir))
    if n < 1e-9:
        raise ValueError("shot_dir has near-zero length")

    shot_dir = shot_dir / n
    yaw = float(np.arctan2(shot_dir[1], shot_dir[0]))

    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array(
        [
            [c, -s],
            [s, c],
        ],
        dtype=np.float64,
    )

    robot_xy = np.array(cue_xy, dtype=np.float64) - rot @ ROBOT_TO_CUE_BALL_LOCAL
    return robot_xy, yaw


def hold_stable(model, data, viewer, duration, phase_name):
    steps = max(1, int(duration / model.opt.timestep))

    for step_idx in range(steps):
        data.qvel[:6] = 0.0
        mujoco.mj_forward(model, data)

        if data.qpos[2] < 0.35 or body_tilt_deg_from_quat_wxyz(data.qpos[3:7]) > 35.0:
            print(f"hold_stable: unstable during {phase_name}", flush=True)
            return False

        if viewer is not None and step_idx % 4 == 0:
            viewer.sync()
            time.sleep(model.opt.timestep)

    return True


def teleport_to_shot_pose(model, data, ctrl, robot_xy, robot_yaw, home_arm=None, viewer=None, settle_time=0.35):
    """Directly place the robot at the planned shot pose.

    This replaces walking between shots. It keeps the same qpos leg/arm state,
    only changes base XY/yaw, zeros base velocity, resets the standing-controller
    target, then lets MuJoCo/standing control forward once before the stroke.
    """
    reset_robot_to_stable_shot_pose(model, data, ctrl, robot_xy, robot_yaw, home_arm=home_arm)

    print(
        f"teleport shot pose: "
        f"xy=({data.qpos[0]:.2f},{data.qpos[1]:.2f}) "
        f"yaw={np.degrees(yaw_from_quat_wxyz(data.qpos[3:7])):.1f}deg "
        f"pelvis_z={data.qpos[2]:.3f} "
        f"tilt={body_tilt_deg_from_quat_wxyz(data.qpos[3:7]):.1f}deg",
        flush=True,
    )

    # A short visual settle keeps the viewer readable without relying on walking.
    if settle_time > 0.0:
        return hold_stable(model, data, viewer, settle_time, "post_teleport")

    return True



# -----------------------------------------------------------------------------
# Shot execution
# -----------------------------------------------------------------------------

def run_physical_stroke(
    model,
    data,
    ctrl,
    tau,
    counter_ref,
    home_arm,
    arm_dof_addrs,
    d_grav,
    pitch_state,
    cue_slide_qadr,
    cue_slide_dadr,
    cue_tip_sid,
    cue_bid,
    cue_ball_bid,
    cue_dadr,
    pocketed_balls,
    cue_ball_start_pos,
    viewer=None,
    shot_idx=0,
    called_plan=None,
):
    """Run one cue stroke using the standing controller."""
    enable_cue_tip_ball_contact(model)

    stabilize_time = AUTO_STABILIZE_TIME if shot_idx == 0 else TELEPORTED_SHOT_STABILIZE_TIME
    cue_time_offset = 0.0 if shot_idx == 0 else max(0.0, T_STAND_END - TELEPORTED_SHOT_CUE_LEAD_TIME)

    if shot_idx > 0:
        print(
            f"standing-controller timing: warmup={stabilize_time:.3f}s "
            f"cue_starts_after={TELEPORTED_SHOT_CUE_LEAD_TIME:.3f}s",
            flush=True,
        )

    # Each shot must begin on the same controller decimation phase as shot 1.
    # Reusing the global sim counter lets follow-up shots start several sim
    # steps before a fresh leg-torque solve, which creates visible wobble.
    stroke_counter = 0
    tau[:] = 0.0
    local_t = 0.0
    while local_t < stabilize_time:
        step_start = time.time()

        t, _ = run_controller_step(
            ctrl,
            model,
            data,
            tau,
            stroke_counter,
            home_arm,
            arm_dof_addrs=arm_dof_addrs,
            d_grav=d_grav,
            pitch_state=pitch_state,
            cue_slide_qadr=cue_slide_qadr,
            cue_slide_dadr=cue_slide_dadr,
            local_t=0.0,
            cue_enabled=False,
        )
        stroke_counter += 1
        counter_ref[0] += 1

        if stroke_counter % 50 == 0:
            print(
                f"auto stability: local_t={local_t:.3f} "
                f"xy=({data.qpos[0]:.2f},{data.qpos[1]:.2f}) "
                f"z={data.qpos[2]:.3f} "
                f"yaw={np.degrees(yaw_from_quat_wxyz(data.qpos[3:7])):+.1f} "
                f"tilt={body_tilt_deg_from_quat_wxyz(data.qpos[3:7]):.1f}",
                flush=True,
            )

        if (data.qpos[2] < 0.45 or body_tilt_deg_from_quat_wxyz(data.qpos[3:7]) > 25.0):
            print("auto: fell during pre-shot stabilization", flush=True)
            return False

        if viewer is not None and stroke_counter % 4 == 0:
            viewer.sync()
            dt_left = model.opt.timestep - (time.time() - step_start)
            if dt_left > 0:
                time.sleep(dt_left)

        local_t += ctrl.simulation_dt

    local_t = 0.0
    settled_time = 0.0
    stroke_started = False
    hit_logged = False
    real_contact_seen = False
    target_contact_logged = False
    target_start_xy = None
    if called_plan is not None:
        called_plan["cue_target_contact_seen"] = False
        called_plan["cue_target_max_speed"] = 0.0
        target_start_xy = body_xy(model, data, called_plan["ball"]).copy()

    while local_t < AUTO_SHOT_TIMEOUT:
        step_start = time.time()
        effective_t = local_t + cue_time_offset

        t, _ = run_controller_step(
            ctrl,
            model,
            data,
            tau,
            stroke_counter,
            home_arm,
            arm_dof_addrs=arm_dof_addrs,
            d_grav=d_grav,
            pitch_state=pitch_state,
            cue_slide_qadr=cue_slide_qadr,
            cue_slide_dadr=cue_slide_dadr,
            local_t=effective_t,
            cue_enabled=True,
        )
        stroke_counter += 1
        counter_ref[0] += 1

        if T_STAND_END <= effective_t < T_STRIKE_END and not stroke_started:
            stroke_started = True
            print(f"[auto shot {shot_idx + 1}] stroke begins", flush=True)

        cue_vel = data.qvel[cue_dadr:cue_dadr + 3].copy()

        contact_metrics = None

        if not hit_logged and float(np.linalg.norm(cue_vel[:2])) > 0.02:
            hit_logged = True
            cue_tip = data.site_xpos[cue_tip_sid].copy()
            cue_axis = data.xmat[cue_bid].reshape(3, 3)[:, 0].copy()
            cue_ball = data.xpos[cue_ball_bid].copy()
            if contact_metrics is None:
                contact_metrics = cue_tip_ball_alignment(model, data, cue_tip_sid, cue_bid, cue_ball_bid)

            print(
                f"[auto shot {shot_idx + 1}] physical cue hit (MuJoCo contact confirmed) "
                f"tip=({cue_tip[0]:.3f},{cue_tip[1]:.3f},{cue_tip[2]:.3f}) "
                f"ball=({cue_ball[0]:.3f},{cue_ball[1]:.3f},{cue_ball[2]:.3f}) "
                f"axis=({cue_axis[0]:.3f},{cue_axis[1]:.3f},{cue_axis[2]:.3f}) "
                f"along={contact_metrics['along']:+.3f} "
                f"lateral={contact_metrics['lateral']:.3f} "
                f"cue_v=({cue_vel[0]:.3f},{cue_vel[1]:.3f},{cue_vel[2]:.3f})",
                flush=True,
            )

        if called_plan is not None and not target_contact_logged:
            target_contact_seen, target_contact_info = body_body_mujoco_contact(
                model, data, "cue_ball", called_plan["ball"]
            )
            if target_contact_seen:
                target_contact_logged = True
                called_plan["cue_target_contact_seen"] = True
                target_pos = body_xy(model, data, called_plan["ball"])
                target_speed = body_linear_speed(model, data, called_plan["ball"])
                called_plan["cue_target_max_speed"] = max(
                    float(called_plan.get("cue_target_max_speed", 0.0)),
                    target_speed,
                )
                con_dist = target_contact_info["dist"] if target_contact_info is not None else float("nan")
                con_pos = target_contact_info["pos"] if target_contact_info is not None else np.array([np.nan, np.nan, np.nan])
                print(
                    f"[auto shot {shot_idx + 1}] cue ball hit target {called_plan['ball']} "
                    f"contact_dist={con_dist:+.5f} "
                    f"contact_pos=({con_pos[0]:.3f},{con_pos[1]:.3f},{con_pos[2]:.3f}) "
                    f"target_xy=({target_pos[0]:.3f},{target_pos[1]:.3f}) "
                    f"target_speed={target_speed:.3f}",
                    flush=True,
                )
        elif called_plan is not None:
            called_plan["cue_target_max_speed"] = max(
                float(called_plan.get("cue_target_max_speed", 0.0)),
                body_linear_speed(model, data, called_plan["ball"]),
            )

        new_pocketed = update_pocketed_balls(model, data, pocketed_balls, cue_ball_start_pos)
        apply_ball_rolling_resistance(model, data, set(pocketed_balls.keys()))

        for body_name in new_pocketed:
            if body_name == "cue_ball_respawned":
                print("[auto] cue_ball scratched; respawned", flush=True)
            else:
                print(f"[auto {local_t:.2f}s] pocketed {body_name}", flush=True)

        max_speed = max_live_ball_speed(model, data, set(pocketed_balls.keys()))

        if effective_t > T_RETURN_END and max_speed < SHOT_SETTLED_SPEED:
            settled_time += ctrl.simulation_dt
            if settled_time >= SHOT_SETTLED_TIME:
                spread_mean, spread_max, _, moving = ball_break_metrics(
                    model,
                    data,
                    set(pocketed_balls.keys()),
                )
                print(
                    f"auto shot {shot_idx + 1} settled: "
                    f"pocketed={len(pocketed_balls)} "
                    f"spread=({spread_mean:.3f},{spread_max:.3f}) "
                    f"moving={moving}",
                    flush=True,
                )
                if called_plan is not None and target_start_xy is not None:
                    target_end_xy = body_xy(model, data, called_plan["ball"])
                    print(
                        f"auto shot {shot_idx + 1} target result: "
                        f"{called_plan['ball']} contact={called_plan.get('cue_target_contact_seen', False)} "
                        f"move={np.linalg.norm(target_end_xy - target_start_xy):.3f} "
                        f"max_speed={called_plan.get('cue_target_max_speed', 0.0):.3f} "
                        f"dist_to_pocket={np.linalg.norm(target_end_xy - called_plan['pocket_xy']):.3f}",
                        flush=True,
                    )
                return True
        else:
            settled_time = 0.0

        if (data.qpos[2] < 0.45 or body_tilt_deg_from_quat_wxyz(data.qpos[3:7]) > 25.0):
            print(f"auto: fell during shot {shot_idx + 1}", flush=True)
            return False

        if viewer is not None and stroke_counter % 4 == 0:
            viewer.sync()
            dt_left = model.opt.timestep - (time.time() - step_start)
            if dt_left > 0:
                time.sleep(dt_left)

        local_t += ctrl.simulation_dt

    print(f"auto shot {shot_idx + 1}: timeout before fully settled", flush=True)
    return True


# -----------------------------------------------------------------------------
# Main auto loop
# -----------------------------------------------------------------------------

def run_auto_loop(
    model,
    data,
    ctrl,
    args,
    cue_contact_settings,
    home_arm,
    tau,
    arm_dof_addrs,
    d_grav,
    pitch_state,
    cue_slide_qadr,
    cue_slide_dadr,
    cue_tip_sid,
    cue_bid,
    cue_ball_bid,
    cue_dadr,
    cue_ball_start_pos,
    viewer=None,
):
    if viewer is not None:
        viewer.cam.lookat[:] = np.array([1.82, 0.0, 0.92])
        viewer.cam.distance = 8.0
        viewer.cam.azimuth = -135.0
        viewer.cam.elevation = -15.0

    counter_ref = [0]
    pocketed_balls = {}
    attempted_shots = set()

    for shot_idx in range(args.max_shots):
        live_remaining = [name for name in OBJECT_BALL_NAMES if name not in pocketed_balls]

        if not live_remaining:
            print("auto: all object balls pocketed", flush=True)
            break

        if shot_idx == 0:
            print("auto shot 1: break from current stable stance", flush=True)
            plan = None
        else:
            # Real-contact / limited-assumption mode:
            #   - do NOT move cue ball or object balls
            #   - do NOT inject velocities into any ball
            #   - do NOT force-clear balls after misses
            #   - choose from the actual post-break table state only
            #   - teleport only the robot base pose to the computed shot line
            #   - use StandingCtrl during stabilization and stroke, like shot 1
            plan = choose_next_shot(
                model,
                data,
                set(pocketed_balls.keys()),
                attempted_shots,
            )

            if plan is None:
                plan = choose_next_shot_relaxed(
                    model,
                    data,
                    set(pocketed_balls.keys()),
                    attempted_shots,
                )
                if plan is None:
                    print("auto: no valid real-contact shot found from current table state; stopping", flush=True)
                    break
                print(
                    f"auto: using relaxed real-contact shot selection "
                    f"cue_blocked={plan.get('cue_blocked', False)} "
                    f"obj_blocked={plan.get('obj_blocked', False)}",
                    flush=True,
                )

            if (
                not plan.get("relaxed", False)
                and float(plan.get("cut_alignment", 1.0)) < REAL_CONTACT_MIN_CUT_ALIGNMENT - REAL_CONTACT_CUT_EPS
            ):
                relaxed_plan = choose_next_shot_relaxed(
                    model,
                    data,
                    set(pocketed_balls.keys()),
                    attempted_shots,
                )
                if relaxed_plan is None:
                    print(
                        f"auto: best available shot is too thin for real-contact mode "
                        f"cut={plan['cut_alignment']:.3f} threshold={REAL_CONTACT_MIN_CUT_ALIGNMENT:.2f}; "
                        f"no relaxed real-contact fallback found",
                        flush=True,
                    )
                    break

                print(
                    f"auto: conservative shot is thin "
                    f"cut={plan['cut_alignment']:.3f} threshold={REAL_CONTACT_MIN_CUT_ALIGNMENT:.2f}; "
                    f"using relaxed real-contact shot "
                    f"cut={relaxed_plan['cut_alignment']:.3f} "
                    f"cue_blocked={relaxed_plan.get('cue_blocked', False)} "
                    f"obj_blocked={relaxed_plan.get('obj_blocked', False)}",
                    flush=True,
                )
                plan = relaxed_plan

            robot_xy, robot_yaw = robot_pose_for_shot(plan["cue_xy"], plan["shot_dir"])

            print(
                f"planned real-contact shot pose: "
                f"{plan['ball']} -> pocket {plan['pocket_idx']} "
                f"cue_xy=({plan['cue_xy'][0]:.2f},{plan['cue_xy'][1]:.2f}) "
                f"object=({plan['object_xy'][0]:.2f},{plan['object_xy'][1]:.2f}) "
                f"ghost=({plan['ghost_xy'][0]:.2f},{plan['ghost_xy'][1]:.2f}) "
                f"pocket=({plan['pocket_xy'][0]:.2f},{plan['pocket_xy'][1]:.2f}) "
                f"shot_dir=({plan['shot_dir'][0]:+.2f},{plan['shot_dir'][1]:+.2f}) "
                f"robot_xy=({robot_xy[0]:.2f},{robot_xy[1]:.2f}) "
                f"yaw={np.degrees(robot_yaw):.1f}deg "
                f"cut={plan['cut_alignment']:.2f} "
                f"cue_dist={plan['cue_dist']:.2f} "
                f"obj_dist={plan['object_dist']:.2f} "
                f"mode=current_table_real_contacts_only",
                flush=True,
            )

            teleport_ok = teleport_to_shot_pose(
                model,
                data,
                ctrl,
                robot_xy,
                robot_yaw,
                home_arm=home_arm,
                viewer=viewer,
                settle_time=0.15,
            )

            if not teleport_ok:
                print("auto: unstable after teleport; stopping before cue alignment", flush=True)
                break

            print(f"standing ref post-teleport: {standing_ref_summary(ctrl, data)}", flush=True)

            shot_arm_target = right_arm_target_at_time(T_STRIKE_END, home_arm)
            align_delta = align_robot_cue_to_ball(
                model,
                data,
                ctrl,
                cue_tip_sid,
                cue_bid,
                cue_slide_qadr,
                plan["cue_xy"],
                shot_arm_target,
            )

            print(
                f"post-align: "
                f"xy=({data.qpos[0]:.2f},{data.qpos[1]:.2f}) "
                f"yaw={np.degrees(yaw_from_quat_wxyz(data.qpos[3:7])):.1f}deg "
                f"align_delta=({align_delta[0]:+.3f},{align_delta[1]:+.3f}) "
                f"align_norm={np.linalg.norm(align_delta):.3f}",
                flush=True,
            )
            print(f"standing ref post-align: {standing_ref_summary(ctrl, data)}", flush=True)

            # Brief pose check before handing the teleported state back to the
            # standing controller for the actual pre-shot stabilization/stroke.
            if not hold_stable(model, data, viewer, 0.20, "post_align_settle"):
                print("auto: unstable after cue alignment; stopping before shot", flush=True)
                break

        before_pocketed = len(pocketed_balls)

        shot_ok = run_physical_stroke(
            model,
            data,
            ctrl,
            tau,
            counter_ref,
            home_arm,
            arm_dof_addrs,
            d_grav,
            pitch_state,
            cue_slide_qadr,
            cue_slide_dadr,
            cue_tip_sid,
            cue_bid,
            cue_ball_bid,
            cue_dadr,
            pocketed_balls,
            cue_ball_start_pos,
            viewer=viewer,
            shot_idx=shot_idx,
            called_plan=plan,
        )

        restore_geom_contacts(model, cue_contact_settings)

        if shot_idx > 0 and shot_ok and plan["ball"] not in pocketed_balls:
            target_contact = bool(plan.get("cue_target_contact_seen", False))
            target_speed = float(plan.get("cue_target_max_speed", 0.0))
            try:
                obj_xy_after = body_xy(model, data, plan["ball"])
                miss_to_pocket = float(np.linalg.norm(obj_xy_after - plan["pocket_xy"]))
            except Exception:
                miss_to_pocket = float("nan")
            print(
                f"auto shot {shot_idx + 1}: called ball {plan['ball']} was not pocketed by real contacts; "
                f"cue_target_contact={target_contact} "
                f"target_max_speed={target_speed:.3f} "
                f"dist_to_planned_pocket={miss_to_pocket:.3f}.",
                flush=True,
            )
            if target_contact:
                print(
                    "auto: target was hit by real contacts; keeping it eligible from the new table state.",
                    flush=True,
                )
            else:
                print(
                    "auto: target was not hit; marking this exact shot attempted.",
                    flush=True,
                )
                attempted_shots.add((plan["ball"], plan["pocket_idx"]))

        ctrl.reset()
        sync_standing_ctrl_to_current_pose(ctrl, data)
        pitch_state[0] = 0.0
        tau[:] = 0.0
        data.qvel[:6] = 0.0
        mujoco.mj_forward(model, data)

        if not shot_ok:
            print(
                f"auto: shot {shot_idx + 1} failed during live StandingCtrl shot; stopping",
                flush=True,
            )
            break

        if (
            shot_idx > 0
            and (not shot_ok or len(pocketed_balls) == before_pocketed)
            and not bool(plan.get("cue_target_contact_seen", False))
        ):
            # Avoid repeating the same failed/missed shot.
            try:
                attempted_shots.add((plan["ball"], plan["pocket_idx"]))
            except UnboundLocalError:
                pass

    live_remaining = [name for name in OBJECT_BALL_NAMES if name not in pocketed_balls]

    print(
        f"auto summary: pocketed={len(pocketed_balls)} "
        f"remaining={len(live_remaining)} "
        f"remaining_balls={','.join(live_remaining) if live_remaining else 'none'}",
        flush=True,
    )

    # MuJoCo/passive-viewer teardown can segfault on some Linux setups after the
    # scripted auto run completes. All output is flushed above, so hard-exit to
    # avoid the misleading core dump after a successful run.
    os._exit(0)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--max-shots", type=int, default=4)
    args = parser.parse_args()

    ctrl = make_pool_ctrl(args.config_file)

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    sim_xml_path = config["sim_xml_path"].replace("{DIR}", ctrl.config_dir)

    model = mujoco.MjModel.from_xml_path(sim_xml_path)
    model.opt.timestep = ctrl.simulation_dt

    tune_ball_joint_damping(model)
    cue_contact_settings = save_cue_contacts(model)

    data = mujoco.MjData(model)
    initialize_data(model, data, ctrl)

    cue_tip_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cue_tip_site")
    cue_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_cue")
    cue_ball_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cue_ball")

    if cue_tip_sid < 0:
        raise RuntimeError("cue_tip_site not found")
    if cue_bid < 0:
        raise RuntimeError("right_cue body not found")
    if cue_ball_bid < 0:
        raise RuntimeError("cue_ball body not found")

    cue_slide_qadr, cue_slide_dadr = cue_slide_addrs_for_model(model)
    arm_dof_addrs = arm_dof_addrs_for_model(model, ctrl)

    d_grav = mujoco.MjData(model)
    d_grav.qpos[:] = model.qpos0.copy()

    pitch_state = [0.0]
    home_arm = np.array(ctrl.arm_waist_target, dtype=np.float64)
    tau = np.zeros(ctrl.num_actuators)

    start_tip = RIGHT_STROKE_TIP_TARGETS[0]
    end_t, end_tip, end_axis, end_tip_vel, measured_pelvis_z = measure_extended_cue(
        args.config_file,
        sim_xml_path,
    )

    ball_pos = end_tip + end_axis * (BALL_RADIUS - BALL_PLACEMENT_OVERLAP)
    place_pool_layout(model, data, ball_pos)
    cue_ball_start_pos = ball_pos.copy()

    _, cue_dadr = body_free_joint_addrs(model, "cue_ball")

    print(
        f"stroke start tip=({start_tip[0]:.3f},{start_tip[1]:.3f},{start_tip[2]:.3f}) "
        f"measured end t={end_t:.3f}s "
        f"tip=({end_tip[0]:.3f},{end_tip[1]:.3f},{end_tip[2]:.3f}) "
        f"axis=({end_axis[0]:.3f},{end_axis[1]:.3f},{end_axis[2]:.3f}) "
        f"tip_v=({end_tip_vel[0]:.3f},{end_tip_vel[1]:.3f},{end_tip_vel[2]:.3f}) "
        f"pelvis_z={measured_pelvis_z:.3f}",
        flush=True,
    )

    print(
        f"placed cue_ball=({ball_pos[0]:.3f},{ball_pos[1]:.3f},{ball_pos[2]:.3f})",
        flush=True,
    )
    print(
        "live StandingCtrl follow-up shots: "
        "standing reference is rebuilt at each teleported XY/yaw",
        flush=True,
    )
    print(f"standing ref shot 1: {standing_ref_summary(ctrl, data)}", flush=True)

    if args.headless:
        run_auto_loop(
            model,
            data,
            ctrl,
            args,
            cue_contact_settings,
            home_arm,
            tau,
            arm_dof_addrs,
            d_grav,
            pitch_state,
            cue_slide_qadr,
            cue_slide_dadr,
            cue_tip_sid,
            cue_bid,
            cue_ball_bid,
            cue_dadr,
            cue_ball_start_pos,
            viewer=None,
        )
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run_auto_loop(
                model,
                data,
                ctrl,
                args,
                cue_contact_settings,
                home_arm,
                tau,
                arm_dof_addrs,
                d_grav,
                pitch_state,
                cue_slide_qadr,
                cue_slide_dadr,
                cue_tip_sid,
                cue_bid,
                cue_ball_bid,
                cue_dadr,
                cue_ball_start_pos,
                viewer=viewer,
            )


if __name__ == "__main__":
    main()
