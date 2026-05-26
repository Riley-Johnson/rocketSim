import numpy as np
from numba import njit
from src.sim.quaternion import body_to_world, world_to_body, quat_derivative
from src.sim.forces.thrust import thrust_force_body
from src.sim.forces.aero import aero_force_world

@njit
def derivatives(t, state, config):
    # unpack state
    #pos = state[0:3]
    vel = state[3:6]
    quat = state[6:10]
    omega = state[10:13]
    mass = state[13]

    # environment
    v_rel = vel - config.wind

    # forces in world frame
    f_g = np.array([0.0, 0.0, -config.g * mass])
    f_t = body_to_world(quat, thrust_force_body(t, config))
    f_d = aero_force_world(v_rel, config)
    f_net = f_g + f_t + f_d

    # torque (all vectors in body frame)
    f_aero_body = world_to_body(quat, f_d)
    tau_aero = np.cross(config.r_cp, f_aero_body)
    tau_thrust = np.cross(config.r_engine, thrust_force_body(t, config))
    tau_damping = -config.k_damp * omega
    torque = tau_aero + tau_thrust + tau_damping

    # derivatives
    a = f_net / mass
    quat_prime = quat_derivative(quat, omega)
    omega_prime = np.linalg.solve(config.I, torque - np.cross(omega, config.I @ omega))

    return np.array([
        vel[0], vel[1], vel[2],
        a[0], a[1], a[2],
        quat_prime[0], quat_prime[1], quat_prime[2], quat_prime[3],
        omega_prime[0], omega_prime[1], omega_prime[2],
        0.0
    ])


