import time
import numpy as np
from flapper import Flapper

f = Flapper(backend_server_ip="192.168.0.2")

while True:
    y = f.get_output_measurement()

    print('[student code, main script] output measurement', y)
    
    f.step(u=np.array([1.0, 2.0, 3.0]))

    time.sleep(1) # for testing purposes only