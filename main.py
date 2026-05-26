from src.launch_config import LaunchConfig
from src.sim_config import SimConfig
from src.sim.simulate import simulate
from src.sim.state0 import State0
from src.motor.parse_eng import parse
from src.plot.plot import plot

launch = LaunchConfig()

sim = SimConfig()
sim.mass = launch.mass
sim.thrust_times, sim.thrust_values, sim.mass_times, sim.mass_values = parse(launch.eng_file)

state0 = State0.from_launch_rail(launch.rail_azimuth, launch.rail_angle, mass=launch.mass)
times, states = simulate(state0.to_array(), sim, t_end=launch.time_max)
plot(times, states)