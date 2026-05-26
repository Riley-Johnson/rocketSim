import numpy as np
from src.sim.quaternion import body_to_world, world_to_body, quat_normalize, quat_derivative

def test_identity():
    q = np.array([1.0, 0, 0, 0])
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(body_to_world(q, v), v)

def test_90deg_rotation():
    # 90 degrees around z-axis should rotate x-hat into y-hat
    q = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
    v = np.array([1.0, 0.0, 0.0])
    result = body_to_world(q, v)
    assert np.allclose(result, [0, 1, 0], atol=1e-10)

def test_roundtrip():
    # body_to_world then world_to_body should recover original
    q = quat_normalize(np.array([1.0, 0.5, 0.3, 0.2]))
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(world_to_body(q, body_to_world(q, v)), v, atol=1e-10)

def test_normalization_drift():
    # integrate a constant omega for many steps, check norm stays at 1
    q = np.array([1.0, 0, 0, 0])
    omega = np.array([0.1, 0.05, 0.02])
    dt = 0.01
    for _ in range(10000):
        dq = quat_derivative(q, omega)
        q = q + dt * dq
        q = quat_normalize(q)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-10
