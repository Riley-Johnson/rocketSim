from pathlib import Path
import numpy as np

_DATA_DIR = Path(__file__).parent.parent.parent / 'data'


def parse(filename: str):
    path = _DATA_DIR / filename
    thrust_times, thrust_values = [], []
    prop_mass = total_mass = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            parts = line.split()
            if prop_mass is None:
                # header: name diameter length delays prop_mass total_mass manufacturer
                prop_mass = float(parts[4])
                total_mass = float(parts[5])
            else:
                thrust_times.append(float(parts[0]))
                thrust_values.append(float(parts[1]))

    thrust_times = np.array(thrust_times, dtype=np.float64)
    thrust_values = np.array(thrust_values, dtype=np.float64)

    # mass decreases linearly from total_mass to dry mass over the burn
    dry_mass = total_mass - prop_mass
    mass_times = np.array([0.0, thrust_times[-1]], dtype=np.float64)
    mass_values = np.array([total_mass, dry_mass], dtype=np.float64)

    return thrust_times, thrust_values, mass_times, mass_values