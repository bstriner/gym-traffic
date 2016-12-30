# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"


from gym_traffic.agents import DQN, EpsilonExplorer
from gym_traffic.runners import SimpleRunner
import gym
from gym.wrappers import Monitor
from tqdm import tqdm
import pandas as pd
from gym_traffic.runners.agent_runner import run_agent

train_env = gym.make('CartPole-v0')
agent = DQN(train_env.observation_space, train_env.action_space, memory_size=4, replay_size=128)
explorer = EpsilonExplorer(agent, epsilon=0.3, decay=2e-5)
path = "output/cartpole/dqn"

print("Q")
agent.Q.summary()
print("training_model")
agent.training_model.summary()


def test_env_func():
    return train_env


runner = SimpleRunner(max_steps_per_episode=1000)

run_agent(runner=runner, agent=explorer, test_agent=agent, train_env=train_env, test_env_func=test_env_func,
          nb_episodes=100, test_nb_episodes=100,
          nb_epoch=25, path=path)
