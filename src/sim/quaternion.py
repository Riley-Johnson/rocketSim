import numpy as np
from numba import njit

# Convention: q = [w, x, y, z], body-to-world rotation

@njit
def quat_normalize(q):
    return q / np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)

@njit
def quat_multiply(p, q):
    # Hamilton product p * q
    pw, px, py, pz = p[0], p[1], p[2], p[3]
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw
    ])

@njit
def quat_conjugate(q):
    # This is basically the inverse
    return np.array([q[0], -q[1], -q[2], -q[3]])

@njit
def body_to_world(q, v):
    # rotate vector v from body frame to world frame
    # using q * [0,v] * q^-1
    qv = np.array([0, v[0], v[1], v[2]])
    result = quat_multiply(quat_multiply(q, qv), quat_conjugate(q))
    return result[1:]   # drop scalar part

@njit
def world_to_body(q, v):
    # reverse rotation
    return body_to_world(quat_conjugate(q), v)

@njit
def quat_derivative(q, omega):
    # omega is angular velocity in body frame, shape (3)
    # returns dq/dt, shape (4)
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
    return 0.5 * quat_multiply(q, omega_quat)

def euler_to_quat(azimuth_deg: float, tilt_deg: float) -> np.ndarray:
    az = np.radians(azimuth_deg)
    el = np.radians(90.0 - tilt_deg)  # tilt from vertical → elevation from horizontal

    # launch direction in world frame (x=East, y=North, z=Up)
    nose = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])

    # quaternion rotating body +z to nose direction
    body_z = np.array([0.0, 0.0, 1.0])
    cross = np.cross(body_z, nose)
    dot = float(np.dot(body_z, nose))
    cross_mag = float(np.linalg.norm(cross))

    if cross_mag < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0]) if dot > 0 else np.array([0.0, 1.0, 0.0, 0.0])

    axis = cross / cross_mag
    half = np.arccos(np.clip(dot, -1.0, 1.0)) / 2.0
    return np.array([np.cos(half), *(axis * np.sin(half))])


@njit
def angle_of_attack(q, vel_world):
    # transform velocity to body frame
    vel_body = world_to_body(q, vel_world)
    v_mag = np.sqrt(vel_body[0]**2 + vel_body[1]**2 + vel_body[2]**2)
    if v_mag < 1e-6:
        return 0.0
    # rocket nose points along body z-axis
    # AoA is angle between velocity and nose axis
    cos_alpha = vel_body[2] / v_mag   # assuming nose = +z in body frame
    return np.arccos(np.clip(cos_alpha, -1.0, 1.0))


