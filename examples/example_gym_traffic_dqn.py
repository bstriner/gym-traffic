#import os

#os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"


import gym
import gym_traffic
from gym_traffic.agents import DQN
from gym_traffic.runners import SimpleRunner
import gym
import pandas as pd
import os

def test(env, agent, path, runner, nb_episodes = 10):
    env.monitor.start(path)
    rewards = runner.run(env, agent, nb_episodes=nb_episodes, render=True, train=False)
    env.monitor.close()
    df = pd.DataFrame({"rewards":rewards})
    df.to_csv(os.path.join(path,"rewards.csv"))

seed = 123
#Create model
train_env = gym.make('Traffic-Simple-v0', mode="cli")
agent = DQN(train_env.observation_space, train_env.action_space)
print "Q"
agent.Q.summary()
print "training_model"
agent.training_model.summary()

#Initial test of model
test_env = gym.make('Traffic-Simple-v0', mode="gui", seed=seed)
test(test_env, agent, "output/simple/epoch-0")

#Train model
nb_episodes=10000
runner = SimpleRunner()
runner.run(train_env, agent, nb_episodes=nb_episodes, render=False, train=True)
agent.save("output/simple/epoch-10000/model.h5")

#Test model again
test_env = gym.make('Traffic-Simple-v0', mode="gui", seed=seed)
test(test_env, agent, "output/simple/epoch-10000")

