import time
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import json
import paho.mqtt.client as mqtt

class Flapper:
    def __init__(self, backend_server_ip=None, robot_pose=None):
        self.SAMPLING_TIME_INTERVAL = 100 # 100 ms
        self.last_time_set_input = int(round(time.time()*1000))
        self.last_time_get_output = int(round(time.time()*1000))

        # mqtt
        self.MQTT_CLIENT_NAME = "student_client"
        self.MQTT_BROKER = backend_server_ip
        self.MQTT_PORT = 1883
        self.MQTT_KEEPALIVE = 60
        self.MQTT_ROBOT_STATE_AND_INPUTS_TOPIC_NAME = "robot_state_and_inputs"
        self.MQTT_ROBOT_POSE_TOPIC_NAME = "robot_pose"
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, self.MQTT_CLIENT_NAME)
        self.mqtt_client.on_connect = lambda client, userdata, flags, rc: self.mqtt_on_connect(client, userdata, flags, rc)
        self.mqtt_client.on_message = lambda client, userdata, msg: self.mqtt_on_message(client, userdata, msg)
        self.mqtt_client.connect(self.MQTT_BROKER, port=self.MQTT_PORT, keepalive=self.MQTT_KEEPALIVE)
        self.mqtt_client.loop_start()

        # initialize state, input, output
        self.x = np.zeros((9,)) # state: position, velocity, acceleration
        self.u = np.zeros((3,))
        self.y = np.zeros((3,))

        self.VERBOSE = False

    def mqtt_on_connect(self, client, userdata, flags, rc):
        print("Connected to the MQTT network with result code " + str(rc))
        self.mqtt_client.subscribe(self.MQTT_ROBOT_POSE_TOPIC_NAME)
        print("Subscribed to the MQTT topic:", self.MQTT_ROBOT_POSE_TOPIC_NAME)

    def mqtt_on_message(self, client, userdata, msg):
        if msg.topic == self.MQTT_ROBOT_POSE_TOPIC_NAME:
            self.robot_pose = json.loads(msg.payload)
            self.y = np.array(self.robot_pose[0:3]).reshape((3,))
        if self.VERBOSE:
            print("[students code, flapper class, on message] self.y", self.y)
    
    def get_output_measurement(self):
        while True:
            delta_time_get_output = int(round(time.time()*1000)) - self.last_time_get_output
            if delta_time_get_output > self.SAMPLING_TIME_INTERVAL:
                self.last_time_get_output = int(round(time.time()*1000))
                print(self.y.shape)
                return self.y
    
    def step(self, x, u):
        delta_time_set_input = int(round(time.time()*1000)) - self.last_time_set_input
        if delta_time_set_input > self.SAMPLING_TIME_INTERVAL:
            self.x = x.reshape((9,))
            self.u = u.reshape((3,))
            self.last_time_set_input = int(round(time.time()*1000))

            robot_state_and_inputs = self.x.tolist() + self.u.tolist()
            
            self.mqtt_client.publish(self.MQTT_ROBOT_STATE_AND_INPUTS_TOPIC_NAME, json.dumps(robot_state_and_inputs))

            if self.VERBOSE:
                print("[students code, flapper class, step] robot_state_and_inputs", robot_state_and_inputs)
