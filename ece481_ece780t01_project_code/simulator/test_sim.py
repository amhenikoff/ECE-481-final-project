import numpy as np
from flapper_sim import FlapperSim

f = FlapperSim(robot_pose=np.array([1.0, 2.0, 0.8, 1.57]))

while True:
    y = f.get_output_measurement()
    
    f.step(u=np.array([1.0, 2.0, 3.0]))
