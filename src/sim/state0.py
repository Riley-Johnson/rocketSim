from dataclasses import dataclass, field
import numpy as np
from src.sim.quaternion import euler_to_quat


@dataclass
class State0:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 10.0

    @classmethod
    def from_launch_rail(cls, azimuth_deg: float, elevation_deg: float, **kwargs) -> 'State0':
        return cls(quat=euler_to_quat(azimuth_deg, elevation_deg), **kwargs)

    def to_array(self) -> np.ndarray:
        return np.concatenate([self.pos, self.vel, self.quat, self.omega, [self.mass]])