import gym
import gym_traffic
from gym_traffic.agents.dqn import DQN
import gym
env = gym.make('Traffic-Simple-v0')
agent = DQN(env.observation_space, env.action_space)
for i_episode in range(10):
    agent.new_episode()
    observation = env.reset()
    for t in range(10000):
        env.render()
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        agent.learn(observation, action, reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break