"""
State:
[x, y, z,
vx, vy, vz,
q0, q1, q2, q3,
w1, w2, w3,
"""

def derivative(t, y):
    """
    Derivative function for the simulation.
    Inputs:
        Time (float): Simulation time
        State (array): Contains all time-specific information for the simulation
        at a certain time step
    Returns:
        State' (array): Derivative of the state at a certain time step
    """

    # Control desired -> rate-limited control

    # Forces - Aero, thrust, gravity

    # F=ma

    # Torques :(

    # rotation shiz

    return y_prime