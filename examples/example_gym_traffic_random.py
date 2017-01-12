import gym
import gym_traffic
from gym.wrappers import Monitor
import gym
import time
#env = gym.make('Traffic-Simple-gui-v0')
from tqdm import tqdm
monitor = False
env = gym.make('Traffic-Simple-cli-v0')
if monitor:
    env = Monitor(env, "output/traffic/simple/random", force=True)
for i_episode in tqdm(range(500)):
    observation = env.reset()
    for t in tqdm(range(1000)):
        #env.render()
        #print(observation)
        action = env.action_space.sample()
        #time.sleep(1)
        observation, reward, done, info = env.step(action)
        #print "Reward: {}".format(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break