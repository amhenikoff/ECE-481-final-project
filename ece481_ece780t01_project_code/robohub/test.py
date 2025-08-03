import time
import numpy as np
from flapper import Flapper

f = Flapper(backend_server_ip="192.168.0.102")

# Logging buffers
position_log = []
target_position_log = []
target_state_log = []
estimated_state_log = []
estimated_state_log_2 = []
u_log = []

# Discrete time state space model of robot
T = 0.1  # Sampling time
A_d = np.array([[1, 0.1, 0.005],
                [0, 1, 0.1],
                [0, 0, 1]])
B_d = np.array([[0.00016666667], [0.005], [0.1]])
C_d = np.array([[1, 0, 0]])

# State feedback controller
F_matrix = 10**2 * np.array([[-1.879614177380125,  -1.069588791089774,  -0.149871934158914]])
# State observer
L = np.array([[-1.529582004419328], [-5.871028337456465], [-9.493798716337043]])

# Target path
P0 = np.array([0, 0, 0.8])
P1 = np.array([1, 0, 0.8])
t0 = 0
t1 = 20

# Target state function
def get_target_x(time):
    if time <= t0:
        output = np.vstack((P0, [0, 0, 0], [0, 0, 0]))
        return output
    elif time > t1:
        output = np.vstack((P1, [0, 0, 0], [0, 0, 0]))
        return output
    elif time <= 0.5 * (t0 + t1):
        pos = 2 * (P1 - P0) * ((time - t0) / (t1 - t0)) ** 2 + P0
        vel = 4 * (P1 - P0) * (time - t0) / (t1 - t0) ** 2
        acc = 4 * (P1 - P0) / (t1 - t0) ** 2
        return np.vstack((pos, vel, acc))
    elif time > 0.5 * (t0 + t1):
        pos = P1 - 2 * (P1 - P0) * ((time - t1) / (t1 - t0)) ** 2
        vel = -4 * (P1 - P0) * (time - t1) / (t1 - t0) ** 2
        acc = -4 * (P1 - P0) / (t1 - t0) ** 2
        return np.vstack((pos, vel, acc)) # [ [pos_x, pos_y, pos_z], [vel_x, vel_y, vel_z], [acc_x, acc_y, acc_z] ]

# Initialize variables
y = f.get_output_measurement()
y = y.reshape((3,1))
y_x_dim = y[0]
y_y_dim = y[1]
y_z_dim = y[2]
# Initial state estimates
x_hat_x_dim = np.array([y[0], [0], [0]])
x_hat_y_dim = np.array([y[1], [0], [0]])
x_hat_z_dim = np.array([y[2], [0], [0]])

# Main Loop
start_time = time.time()
while time.time() - start_time <= 20:
    t = time.time() - start_time

    # Measure y
    y = f.get_output_measurement()
    y = y.reshape((3,1))
    y_x_dim = y[0]
    y_y_dim = y[1]
    y_z_dim = y[2]

    print(y)

    # Get target state
    target_x = get_target_x(t)

    # Controller
    u_x_dim = F_matrix @ (x_hat_x_dim - np.expand_dims(target_x[:, 0], axis=-1))
    u_y_dim = F_matrix @ (x_hat_y_dim - np.expand_dims(target_x[:, 1], axis=-1))
    u_z_dim = F_matrix @ (x_hat_z_dim - np.expand_dims(target_x[:, 2], axis=-1))
    u = np.array([u_x_dim[0, 0], u_y_dim[0, 0], u_z_dim[0, 0]])

    # Observer
    x_hat_x_dim = ((A_d + (L @ C_d)) @ x_hat_x_dim) + (B_d * u_x_dim) - (L * y_x_dim)
    x_hat_y_dim = ((A_d + (L @ C_d)) @ x_hat_y_dim) + (B_d * u_y_dim) - (L * y_y_dim)
    x_hat_z_dim = ((A_d + (L @ C_d)) @ x_hat_z_dim) + (B_d * u_z_dim) - (L * y_z_dim)
    x_hat = np.reshape(np.concatenate((x_hat_x_dim, x_hat_y_dim, x_hat_z_dim)), (9, 1))
    
    # Step the simulator
    f.step(x=x_hat, u=u)

    # Logging
    position_log.append(y)
    target_position_log.append(([target_x[0,0]], [target_x[0,1]], [target_x[0,2]]))
    target_state_log.append(target_x)
    estimated_state_log.append(([x_hat_x_dim[0, 0]], [x_hat_y_dim[0, 0]], [x_hat_z_dim[0, 0]]))
    estimated_state_log_2.append(np.column_stack((x_hat_x_dim, x_hat_y_dim, x_hat_z_dim)))
    u_log.append(u)

# Save logs to file
position_log = np.array(position_log)
target_position_log = np.array(target_position_log)
target_state_log = np.array(target_state_log)
estimated_state_log = np.array(estimated_state_log)
estimated_state_log_2 = np.array(estimated_state_log_2)
u_log = np.array(u_log)

np.savez('data_log.npz', array_a=position_log, 
                        array_b=target_position_log, 
                        array_c=target_state_log,
                        array_d=estimated_state_log,
                        array_e=estimated_state_log_2,
                        array_f=u_log)