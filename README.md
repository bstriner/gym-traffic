# gym-traffic
OpenAI Gym Environment for Traffic Control

## Installation
```buildoutcfg
git clone https://github.com/bstriner/gym-traffic.git
cd gym-traffic
python setup.py install
```

##Environments

###Simple Environment
`Traffic-Simple-cli-v0` and `Traffic-Simple-gui-v0` model a simple intersection 
with North-South, South-North, East-West, and West-East traffic.

CLI runs `sumo` and GUI runs `sumo-gui`. GUI is slower but required if you want to render video.

Agent has 4 available actions, corresponding to traffic light phases:
* Green N-S Red E-W
* Yellow N-S Red E-W
* Red N-S Green E-W
* Red N-S Yellow E-W

![Simple Environment](https://github.com/bstriner/gym-traffic/raw/master/doc/images/simple-env.png)

##Agents

A simple DQN agent is provided, written in Keras.
* [DQN Agent](https://github.com/bstriner/gym-traffic/blob/master/gym_traffic/agents/dqn.py)
* [DQN Example](https://github.com/bstriner/gym-traffic/blob/master/examples/example_gym_traffic_dqn.py)

##Questions?

Feel free to create issues, pull requests, or email me.
