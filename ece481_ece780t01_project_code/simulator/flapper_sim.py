import time
import math
import random
import matplotlib.pyplot as plt
import numpy as np

class FlapperSim:
    def __init__(self, backend_server_ip=None, robot_pose=None):
        self.ROBOT_H = 0.4 # 40 cm
        self.ROBOT_W = 0.5 # 50 cm
        self.POS_BOUNDS = [-1.0, 2.0, -2.0, 2.0, self.ROBOT_H, 1.5]
        self.VEL_BOUNDS = np.array([0.1, 0.1, 0.1])
        self.NOISE_STD_DEV = 0.005 # 5 mm
        self.SAMPLING_TIME_INTERVAL = 100 # 100 ms
        self.last_time_set_input = int(round(time.time()*1000))
        self.last_time_get_output = int(round(time.time()*1000))

        # initialize state, input, output
        self.x = np.zeros((9, 1)) # state: position, velocity, acceleration
        if robot_pose is not None:
            self.x[0:3] = np.array(robot_pose[0:3]).reshape((3, 1))
            self.yaw = robot_pose[3]
        else:
            self.x[0:3] = np.array([self.POS_BOUNDS[0] + (self.POS_BOUNDS[1] - self.POS_BOUNDS[0]) * random.random(),
                                self.POS_BOUNDS[2] + (self.POS_BOUNDS[3] - self.POS_BOUNDS[2]) * random.random(),
                                self.POS_BOUNDS[4] + (self.POS_BOUNDS[5] - self.POS_BOUNDS[4]) * random.random()]).reshape((3, 1))
            self.yaw = 2 * math.pi * random.random()
        self.u = np.zeros((3, 1)) # input: jerk
        self.y = self.x[0:3] + np.random.normal(np.zeros((3, 1)), self.NOISE_STD_DEV * np.ones((3, 1)))
                        
        # variables for plotting
        self.robot_body_0 = np.array([[0., 0., 0.2 * self.ROBOT_H],
                                    [0., 0., self.ROBOT_H],
                                    [0., 0., 0.5 * self.ROBOT_H],
                                    [0., 0., 0.75 * self.ROBOT_H],
                                    [self.ROBOT_W / 2. * math.sin(math.pi / 12.), self.ROBOT_W / 2. * math.cos(math.pi / 12.), 0.75 * self.ROBOT_H],
                                    [-self.ROBOT_W / 2. * math.sin(math.pi / 12.), self.ROBOT_W / 2. * math.cos(math.pi / 12.), 0.75 * self.ROBOT_H],
                                    [self.ROBOT_W / 2. * math.sin(math.pi / 12.), -self.ROBOT_W / 2. * math.cos(math.pi / 12.), 0.75 * self.ROBOT_H],
                                    [-self.ROBOT_W / 2. * math.sin(math.pi / 12.), -self.ROBOT_W / 2. * math.cos(math.pi / 12.), 0.75 * self.ROBOT_H],
                                    [self.ROBOT_W / 4., self.ROBOT_W / 4., 0.],
                                    [self.ROBOT_W / 4., -self.ROBOT_W / 4., 0.],
                                    [-self.ROBOT_W / 4., self.ROBOT_W / 4., 0.],
                                    [-self.ROBOT_W / 4., -self.ROBOT_W / 4., 0.]]).T
        
        self.robot_body_segments = [[0, 1],
                                    [3, 4],
                                    [3, 5],
                                    [3, 6],
                                    [3, 7],
                                    [2, 4],
                                    [2, 5],
                                    [2, 6],
                                    [2, 7],
                                    [0, 8],
                                    [0, 9],
                                    [0, 10],
                                    [0, 11]]
        
        # init plot
        self.figure = []
        self.axes = []
        self.handles = []
        self.__init_plot()
    
    def rotate_and_translate_body(self):
        R = np.array([[math.cos(self.yaw), -math.sin(self.yaw), 0.], [math.sin(self.yaw), math.cos(self.yaw), 0.], [0., 0., 1.]])
        t = np.array(self.x[0:3]).reshape((3, 1))
        t[2] -= self.ROBOT_H
        self.robot_body = R @ self.robot_body_0 + np.tile(t, (1, self.robot_body_0.shape[1]))
    
    def __init_plot(self):
        self.figure = plt.figure()
        self.axes = self.figure.add_subplot(projection='3d')
        self.figure.set_size_inches(1200 / plt.rcParams['figure.dpi'], 1200 / plt.rcParams['figure.dpi'])
        self.axes.plot([self.POS_BOUNDS[i] for i in [0, 0, 1, 1, 0]], [self.POS_BOUNDS[i] for i in [2, 3, 3, 2, 2]], [self.POS_BOUNDS[i] for i in [4, 4, 4, 4, 4]], 'k')
        self.axes.plot([self.POS_BOUNDS[i] for i in [0, 0, 1, 1, 0]], [self.POS_BOUNDS[i] for i in [2, 3, 3, 2, 2]], [self.POS_BOUNDS[i] for i in [5, 5, 5, 5, 5]], 'k')
        self.axes.plot([self.POS_BOUNDS[i] for i in [0, 0]], [self.POS_BOUNDS[i] for i in [2, 2]], [self.POS_BOUNDS[i] for i in [4, 5]], 'k')
        self.axes.plot([self.POS_BOUNDS[i] for i in [0, 0]], [self.POS_BOUNDS[i] for i in [3, 3]], [self.POS_BOUNDS[i] for i in [4, 5]], 'k')
        self.axes.plot([self.POS_BOUNDS[i] for i in [1, 1]], [self.POS_BOUNDS[i] for i in [3, 3]], [self.POS_BOUNDS[i] for i in [4, 5]], 'k')
        self.axes.plot([self.POS_BOUNDS[i] for i in [1, 1]], [self.POS_BOUNDS[i] for i in [2, 2]], [self.POS_BOUNDS[i] for i in [4, 5]], 'k')

        self.rotate_and_translate_body()

        for s in self.robot_body_segments:
            i = s[0]
            j = s[1]
            self.handles.append(self.axes.plot([self.robot_body[0, k] for k in [i, j]], [self.robot_body[1, k] for k in [i, j]], [self.robot_body[2, k] for k in [i, j]], 'k')[0])
        
        self.axes.set_xlim(self.POS_BOUNDS[0], self.POS_BOUNDS[1])
        self.axes.set_ylim(self.POS_BOUNDS[2], self.POS_BOUNDS[3])
        self.axes.set_zlim(0., self.POS_BOUNDS[5])
        self.axes.view_init(elev=15., azim=-135., roll=0.)
        self.axes.set_xlabel('x')
        self.axes.set_ylabel('y')
        self.axes.set_zlabel('z')
        self.axes.set_aspect('equal', adjustable='box')

        plt.ion()
        plt.show()
    
    def __update_plot(self):
        self.rotate_and_translate_body()
        
        for idx, s in enumerate(self.robot_body_segments):
            i = s[0]
            j = s[1]
            xy_data = np.array([[self.robot_body[0, k] for k in [i, j]], [self.robot_body[1, k] for k in [i, j]]])
            z_data = [self.robot_body[2, k] for k in [i, j]]
            self.handles[idx].set_data(xy_data)
            self.handles[idx].set_3d_properties(z_data)
        
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
    
    def get_output_measurement(self):
        delta_time_get_output = int(round(time.time()*1000)) - self.last_time_get_output
        if delta_time_get_output > self.SAMPLING_TIME_INTERVAL:
            self.y = self.x[0:3] + np.random.normal(np.zeros((3, 1)), self.NOISE_STD_DEV * np.ones((3, 1)))
            self.last_time_get_output = int(round(time.time()*1000))
        return self.y
    
    def step(self, u):
        delta_time_set_input = int(round(time.time()*1000)) - self.last_time_set_input
        if delta_time_set_input > self.SAMPLING_TIME_INTERVAL:
            self.u = u
            self.last_time_set_input = int(round(time.time()*1000))
        
        # integrate (move the robot)
        dt = delta_time_set_input / 1000.0

        self.x[0] = self.x[0] + self.x[3] * dt
        self.x[1] = self.x[1] + self.x[4] * dt
        self.x[2] = self.x[2] + self.x[5] * dt
        self.x[3] = self.x[3] + self.x[6] * dt
        self.x[4] = self.x[4] + self.x[7] * dt
        self.x[5] = self.x[5] + self.x[8] * dt
        self.x[6] = self.x[6] + self.u[0] * dt
        self.x[7] = self.x[7] + self.u[1] * dt
        self.x[8] = self.x[8] + self.u[2] * dt
        
        # velocity, and corresponding acceleration, bounds
        if self.x[3] < -self.VEL_BOUNDS[0]:
            self.x[3] = -self.VEL_BOUNDS[0]
            self.x[6] = 0.
        elif self.x[3] > self.VEL_BOUNDS[0]:
            self.x[3] = self.VEL_BOUNDS[0]
            self.x[6] = 0.
        
        if self.x[4] < -self.VEL_BOUNDS[1]:
            self.x[4] = -self.VEL_BOUNDS[1]
            self.x[7] = 0.
        elif self.x[4] > self.VEL_BOUNDS[1]:
            self.x[4] = self.VEL_BOUNDS[1]
            self.x[7] = 0.
        
        if self.x[5] < -self.VEL_BOUNDS[2]:
            self.x[5] = -self.VEL_BOUNDS[2]
            self.x[8] = 0.
        elif self.x[5] > self.VEL_BOUNDS[2]:
            self.x[5] = self.VEL_BOUNDS[2]
            self.x[8] = 0.

        # position bounds
        if self.x[0] < self.POS_BOUNDS[0]:
            self.x[3] = 0.01
        elif self.x[0] > self.POS_BOUNDS[1]:
            self.x[3] = -0.01
        
        if self.x[1] < self.POS_BOUNDS[2]:
            self.x[4] = 0.01
        elif self.x[1] > self.POS_BOUNDS[3]:
            self.x[4] = -0.01
        
        if self.x[2] < self.POS_BOUNDS[4]:
            self.x[5] = 0.
        elif self.x[2] > self.POS_BOUNDS[5]:
            self.x[5] = -0.01
        
        self.yaw = self.yaw + (-1 * self.yaw) * dt

        # update plot
        self.__update_plot()
