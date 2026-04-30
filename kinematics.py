# -*- coding: utf-8 -*-
"""
6-Axis Industrial Robot Kinematics
Modified Craig DH Convention
Units: mm, degrees
"""

import numpy as np

# =============================================================================
# DH Parameters - modify these values to match your robot
# Format: [alpha_{i-1} (deg), a_{i-1} (mm), d_i (mm), theta_offset (deg)]
# =============================================================================
DH_PARAMS = [
    [  0.0,   0.0, 100.0,  0.0],   # Joint 1
    [ 90.0,   0.0,   0.0,  0.0],   # Joint 2
    [  0.0, 100.0,   0.0,  0.0],   # Joint 3
    [ 90.0, 100.0,   0.0,  0.0],   # Joint 4
    [-90.0,   0.0,   0.0,  0.0],   # Joint 5
    [ 90.0,   0.0, 100.0,  0.0],   # Joint 6
]

# Joint limits (deg)
JOINT_LIMITS = [
    (-170.0,  170.0),   # J1
    ( -90.0,  135.0),   # J2
    (-180.0,   70.0),   # J3
    (-190.0,  190.0),   # J4
    (-120.0,  120.0),   # J5
    (-360.0,  360.0),   # J6
]


def dh_matrix(alpha_prev, a_prev, d, theta):
    """Modified Craig DH transformation matrix."""
    ca = np.cos(alpha_prev)
    sa = np.sin(alpha_prev)
    ct = np.cos(theta)
    st = np.sin(theta)
    return np.array([
        [ct,      -st,       0,     a_prev  ],
        [st*ca,    ct*ca,   -sa,   -sa*d    ],
        [st*sa,    ct*sa,    ca,    ca*d    ],
        [0,        0,        0,     1       ],
    ])


def forward_kinematics(joint_angles_deg):
    """
    Input:  joint_angles_deg - list of 6 joint angles (deg)
    Output: T      - 4x4 end-effector homogeneous transform (mm)
            T_list - list of cumulative transforms per joint
    """
    T = np.eye(4)
    T_list = [T.copy()]
    for i, q_deg in enumerate(joint_angles_deg):
        alpha = np.deg2rad(DH_PARAMS[i][0])
        a     = DH_PARAMS[i][1]
        d     = DH_PARAMS[i][2]
        theta = np.deg2rad(q_deg + DH_PARAMS[i][3])
        T = T @ dh_matrix(alpha, a, d, theta)
        T_list.append(T.copy())
    return T, T_list


def _rot_to_euler_zyx(R):
    """Rotation matrix to ZYX Euler angles (rad). For display only."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        return np.array([
            np.arctan2( R[1, 0],  R[0, 0]),
            np.arctan2(-R[2, 0],  sy),
            np.arctan2( R[2, 1],  R[2, 2]),
        ])
    else:
        return np.array([
            np.arctan2(-R[1, 2],  R[1, 1]),
            np.arctan2(-R[2, 0],  sy),
            0.0,
        ])


def _euler_zyx_to_rot(rz, ry, rx):
    """ZYX Euler (rad) to rotation matrix."""
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    return Rz @ Ry @ Rx


def _rot_error_axis_angle(R_target, R_current):
    """
    Orientation error as axis-angle vector (rad).
    Returns 3-vector: rotation needed to go from R_current to R_target.
    Avoids Euler angle gimbal-lock issues for IK.
    """
    R_err = R_target @ R_current.T
    cos_t = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-8:
        return np.zeros(3)
    sin_t = np.sin(theta)
    if abs(sin_t) < 1e-8:
        # 180-degree rotation - find principal axis
        # Use diagonal to recover axis
        diag = np.diag(R_err)
        i = int(np.argmax(diag))
        v = R_err[:, i] + np.eye(3)[:, i]
        v = v / max(np.linalg.norm(v), 1e-12)
        return theta * v
    axis = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ]) / (2.0 * sin_t)
    return theta * axis


def get_end_effector_pose(joint_angles_deg):
    """
    Input:  joint_angles_deg - list of 6 joint angles (deg)
    Output: position    - [x, y, z] mm
            euler_zyx   - [rz, ry, rx] deg
    """
    T, _ = forward_kinematics(joint_angles_deg)
    position  = T[:3, 3]
    euler_zyx = np.rad2deg(_rot_to_euler_zyx(T[:3, :3]))
    return position, euler_zyx


def _geometric_jacobian(joint_angles_deg, delta=0.01):
    """
    6x6 geometric Jacobian.
    Top 3 rows: linear velocity (mm per deg of joint).
    Bottom 3 rows: angular velocity in axis-angle form (rad per deg of joint).
    """
    q     = np.array(joint_angles_deg, dtype=float)
    T0, _ = forward_kinematics(q)
    p0    = T0[:3, 3]
    R0    = T0[:3, :3]
    J     = np.zeros((6, 6))
    for i in range(6):
        dq     = q.copy()
        dq[i] += delta
        Ti, _  = forward_kinematics(dq)
        # Linear part
        J[:3, i] = (Ti[:3, 3] - p0) / delta
        # Angular part via axis-angle of R_new * R_old^T
        R_new = Ti[:3, :3]
        w     = _rot_error_axis_angle(R_new, R0)
        J[3:, i] = w / delta
    return J


def inverse_kinematics(target_pos, target_euler_zyx_deg,
                        q_init_deg=None, max_iter=1000,
                        tol_pos=0.1, tol_rot=0.5):
    """
    Numerical IK using Damped Least Squares.
    Uses axis-angle for orientation error to avoid gimbal lock.
    Input:  target_pos           - [x, y, z] mm
            target_euler_zyx_deg - [rz, ry, rx] deg
            q_init_deg           - initial joint angles (deg), default zeros
    Output: q_result - 6 joint angles (deg)
            success  - bool
            error    - (pos_err_mm, rot_err_deg)
    """
    if q_init_deg is None:
        q_init_deg = [0.0] * 6

    q = np.array(q_init_deg, dtype=float)

    # Nudge away from singular zero pose
    if np.allclose(q, 0.0):
        q = q + 1.0

    target_pos = np.array(target_pos, dtype=float)
    rz, ry, rx = np.deg2rad(np.array(target_euler_zyx_deg, dtype=float))
    R_target   = _euler_zyx_to_rot(rz, ry, rx)

    lam      = 0.1
    pos_err  = rot_err = 1e9

    for _ in range(max_iter):
        T_cur, _ = forward_kinematics(q)
        p_cur    = T_cur[:3, 3]
        R_cur    = T_cur[:3, :3]

        e_pos    = target_pos - p_cur
        e_rot    = _rot_error_axis_angle(R_target, R_cur)

        pos_err  = float(np.linalg.norm(e_pos))
        rot_err  = float(np.rad2deg(np.linalg.norm(e_rot)))

        if pos_err < tol_pos and rot_err < tol_rot:
            for i in range(6):
                lo, hi = JOINT_LIMITS[i]
                q[i] = np.clip(q[i], lo, hi)
            return q, True, (pos_err, rot_err)

        e = np.concatenate([e_pos, e_rot])
        J = _geometric_jacobian(q)

        # Damped least squares: dq_deg = J^T (J J^T + lam^2 I)^-1 e
        JJT = J @ J.T
        try:
            dq_deg = J.T @ np.linalg.solve(JJT + lam**2 * np.eye(6), e)
        except np.linalg.LinAlgError:
            dq_deg = J.T @ np.linalg.lstsq(JJT + lam**2 * np.eye(6), e, rcond=None)[0]

        # Limit step size to avoid overshooting past local minima
        max_step = 10.0  # max degrees per joint per iteration
        max_dq   = float(np.max(np.abs(dq_deg)))
        if max_dq > max_step:
            dq_deg = dq_deg * (max_step / max_dq)

        q += dq_deg

        for i in range(6):
            lo, hi = JOINT_LIMITS[i]
            q[i] = np.clip(q[i], lo, hi)

    return q, False, (pos_err, rot_err)


def inverse_kinematics_multistart(target_pos, target_euler_zyx_deg,
                                   q_init_deg=None, n_restarts=8):
    """
    Multi-start IK to improve convergence.
    Tries given initial guess plus random restarts, returns best solution.
    """
    best_q, best_err = None, (1e9, 1e9)
    candidates = [q_init_deg] if q_init_deg is not None else [[0.0] * 6]

    rng = np.random.default_rng(42)
    for _ in range(n_restarts):
        candidates.append([rng.uniform(lo, hi) for lo, hi in JOINT_LIMITS])

    for q0 in candidates:
        q_res, success, err = inverse_kinematics(
            target_pos, target_euler_zyx_deg, q_init_deg=q0
        )
        if success:
            return q_res, True, err
        if err[0] + err[1] < best_err[0] + best_err[1]:
            best_q, best_err = q_res, err

    return best_q, False, best_err


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    q_test = [0.0, -30.0, 60.0, 0.0, 30.0, 0.0]
    pos, euler = get_end_effector_pose(q_test)
    print(f"FK  pos  : {pos.round(2)}")
    print(f"FK  euler: {euler.round(2)}")

    q_res, ok, err = inverse_kinematics_multistart(pos, euler, q_init_deg=[0.0]*6)
    print(f"IK  result : {q_res.round(4)}")
    print(f"IK  success: {ok}  pos_err={err[0]:.4f}mm  rot_err={err[1]:.4f}deg")