from dataclasses import dataclass


@dataclass
class LaunchConfig:
    rail_azimuth: float = 50.0   # compass bearing, degrees (0=North, 90=East)
    rail_angle: float   = 5.0    # degrees from vertical
    rail_length: float  = 15.0   # metres
    mass: float         = 0.5    # initial rocket mass, kg
    time_max: float     = 30.0   # max simulation time, seconds
    eng_file: str       = 'AeroTech_F42T_L.eng'