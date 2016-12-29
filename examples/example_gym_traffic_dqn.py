# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"


from gym_traffic.agents import DQN
from gym_traffic.runners import SimpleRunner
import gym
from gym.wrappers import Monitor
from tqdm import tqdm
import pandas as pd
from gym_traffic.runners.agent_runner import run_agent

train_env = gym.make('Traffic-Simple-cli-v0')
agent = DQN(train_env.observation_space, train_env.action_space)
path = "output/traffic/simple"


def test_env_func():
    return gym.make('Traffic-Simple-gui-v0')


run_agent(agent=agent, train_env=train_env, test_env_func=test_env_func, nb_episodes=500, nb_epoch=50, path=path)
