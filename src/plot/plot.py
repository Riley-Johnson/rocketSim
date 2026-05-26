import matplotlib.pyplot as plt
import numpy as np

def plot(times, states):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].plot(times, states[:, 2])
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Altitude (m)')
    axes[0, 0].set_title('Altitude')

    axes[0, 1].plot(times, np.linalg.norm(states[:, 3:6], axis=1))
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Speed (m/s)')
    axes[0, 1].set_title('Speed')

    axes[1, 0].plot(states[:, 0], states[:, 1])
    axes[1, 0].set_xlabel('X (m)')
    axes[1, 0].set_ylabel('Y (m)')
    axes[1, 0].set_title('Ground Track')
    axes[1, 0].set_aspect('equal')

    axes[1, 1].plot(times, np.linalg.norm(states[:, 10:13], axis=1))
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Angular speed (rad/s)')
    axes[1, 1].set_title('Angular Velocity')

    plt.tight_layout()
    plt.show()