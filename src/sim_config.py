from numba import float64
from numba.experimental import jitclass
import numpy as np

# Numba jitclass spec — every field must be listed here with its type and
# initialized in __init__. Type cheatsheet:
#   scalar:   float64
#   1D array: float64[:]
#   2D array: float64[:, :]
# All fields must be assigned in __init__ (even placeholders),
# because jitclass runs in nopython mode and cannot do lazy initialization.
# Fields sourced from external data (e.g. .eng parse) should be
# initialized to np.empty(0) and set from outside before simulate().

spec = [
    ('g',             float64),
    ('rho',           float64),
    ('wind',          float64[:]),
    ('Cd',            float64),
    ('A',             float64),
    ('r_cp',          float64[:]),
    ('r_engine',      float64[:]),
    ('k_damp',        float64),
    ('I',             float64[:, :]),
    ('mass',          float64),
    ('thrust_times',  float64[:]),
    ('thrust_values', float64[:]),
    ('mass_times',    float64[:]),
    ('mass_values',   float64[:]),
]

@jitclass(spec)
class SimConfig:
    def __init__(self):
        self.g        = 9.81
        self.rho      = 1.225
        self.wind     = np.array([1.0, 1.0, 0.0])
        self.Cd       = 0.45
        self.A        = np.pi * (0.028 ** 2)
        self.r_cp     = np.array([0.0, 0.0, -0.3])
        self.r_engine = np.array([0.0, 0.0, -0.5])
        self.k_damp   = 0.01
        self.I        = np.diag(np.array([0.05, 0.05, 0.002]))
        self.mass     = 0.5

        # placeholders - set from outside before calling simulate()
        self.thrust_times  = np.empty(0)
        self.thrust_values = np.empty(0)
        self.mass_times    = np.empty(0)
        self.mass_values   = np.empty(0)