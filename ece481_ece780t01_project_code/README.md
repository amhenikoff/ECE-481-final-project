# Real robot in the Robohub

Tested with Python 3.12 on Ubuntu 24.04. In the following it is assumed `python3` points to your python executable.

1. Create and activate a Python virtual environment:
```
python3 -m virtualenv ece481_ece780t01_project_venv
source ece481_ece780t01_project_venv/bin/activate
```
This step requires the package `virtualenv` to be installed, which can be done via `python3 -m pip install virtualenv`.

2. Install latest release of the MQTT client package, required to communicate with the backend interfacing with the robots, in order to send commands and receive poses:
```
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade paho-mqtt
```

3. Install numpy and control, required to develop the controller for the flapping wing robot:
```
python3 -m pip install numpy control
```

4. Connect to the following Wi-Fi network:
    * SSID: brushbotarium
    * Password: brushbotarium

5. Run `test.py` to test the communication with the robot:
```
python3 test.py
```

# Simulator

1. Create and activate a Python virtual environment:
```
python3 -m virtualenv ece481_ece780t01_project_sim_venv
source ece481_ece780t01_project_sim_venv/bin/activate
```

2. Install numpy, control, and matplotlib, required to run the simulator and develop the controller:
```
python3 -m pip install numpy control matplotlib PyQt5
```

3. Open the script `test_sim.py` and uncomment either line 6 or 8 in order to instantiate a simulator with random or specified, respectively, poses of the robot.

4. Run `test_sim.py` to test the simulator:
```
python3 test_sim.py
```