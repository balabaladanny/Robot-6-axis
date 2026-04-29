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
    """Rotation matrix to ZYX Euler angles (rad)."""
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


def _jacobian(joint_angles_deg, delta=1e-4):
    """Numerical Jacobian (6x6)."""
    q   = np.array(joint_angles_deg, dtype=float)
    T0, _ = forward_kinematics(q)
    p0  = T0[:3, 3]
    e0  = _rot_to_euler_zyx(T0[:3, :3])
    J   = np.zeros((6, 6))
    for i in range(6):
        dq     = q.copy()
        dq[i] += np.rad2deg(delta)
        Ti, _  = forward_kinematics(dq)
        J[:3, i] = (Ti[:3, 3] - p0) / delta
        J[3:, i] = (_rot_to_euler_zyx(Ti[:3, :3]) - e0) / delta
    return J


def inverse_kinematics(target_pos, target_euler_zyx_deg,
                        q_init_deg=None, max_iter=200,
                        tol_pos=0.1, tol_rot=0.01):
    """
    Numerical IK using Damped Least Squares Jacobian pseudo-inverse.
    Input:  target_pos           - [x, y, z] mm
            target_euler_zyx_deg - [rz, ry, rx] deg
            q_init_deg           - initial joint angles (deg), default zeros
    Output: q_result - 6 joint angles (deg)
            success  - bool
            error    - (pos_err_mm, rot_err_deg)
    """
    if q_init_deg is None:
        q_init_deg = [0.0] * 6

    q            = np.array(q_init_deg, dtype=float)
    target_pos   = np.array(target_pos, dtype=float)
    target_euler = np.deg2rad(np.array(target_euler_zyx_deg, dtype=float))
    lam          = 0.5
    pos_err = rot_err = 1e9

    for _ in range(max_iter):
        T_cur, _ = forward_kinematics(q)
        p_cur    = T_cur[:3, 3]
        e_cur    = _rot_to_euler_zyx(T_cur[:3, :3])

        e_pos    = target_pos - p_cur
        e_rot    = target_euler - e_cur
        pos_err  = float(np.linalg.norm(e_pos))
        rot_err  = float(np.linalg.norm(np.rad2deg(e_rot)))

        if pos_err < tol_pos and rot_err < tol_rot:
            for i in range(6):
                lo, hi = JOINT_LIMITS[i]
                q[i] = np.clip(q[i], lo, hi)
            return q, True, (pos_err, rot_err)

        e   = np.concatenate([e_pos, e_rot])
        J   = _jacobian(q)
        JJT = J @ J.T
        dq  = J.T @ np.linalg.solve(JJT + lam**2 * np.eye(6), e)
        q  += np.rad2deg(dq)

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
    best_q, best_success, best_err = None, False, (1e9, 1e9)
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