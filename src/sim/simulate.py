import numpy as np
from src.sim.integrator import rk4_step

def simulate(state0, config, dt=0.01, t_end=30.0):
    t = 0.0
    state = state0.copy()

    times = [t]
    states = [state.copy()]

    while t < t_end:
        state = rk4_step(t, state, dt, config)
        t += dt
        times.append(t)
        states.append(state.copy())

        # termination condition
        if state[2] < 0 and t > 1.0:  # z < 0, past launch
            break

    return np.array(times), np.array(states)