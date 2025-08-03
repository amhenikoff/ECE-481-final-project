import time
import numpy as np
from flapper_sim import FlapperSim

f = FlapperSim(robot_pose=np.array([1.0, 2.0, 0.8, 1.57]))

T = 0.1

while True:
    time_loop_start = time.time()

    y = f.get_output_measurement()
    
    estimated_state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    control_input = np.array([1.0, 2.0, 3.0])

    f.step(x=estimated_state, u=control_input)

    time_loop_end = time.time()
    time.sleep(max(0, T - (time_loop_end - time_loop_start)))