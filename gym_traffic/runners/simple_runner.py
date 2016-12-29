import numpy as np
from tqdm import tqdm

class SimpleRunner(object):
    def __init__(self, max_steps_per_episode = 1000):
        self.max_steps_per_episode=max_steps_per_episode

    def run(self, env, agent, nb_episodes = 100, render=True, verbose=True, train=True):
        episode_rewards = []
        losses = []
        for episode in tqdm(range(nb_episodes)):
            episode_losses = []
            agent.new_episode()
            observation = env.reset()
            agent.observe(observation)
            total_reward = 0.0
            for t in range(self.max_steps_per_episode):
                if render:
                    env.render()
                action = agent.act()
                observation, reward, done, info = env.step(action)
                #print "Reward: {}, Observation: {}, Done: {}".format(reward, observation, done)
                total_reward += reward
                agent.observe(observation)
                if train:
                    loss = agent.learn(action, reward, done)
                    episode_losses.append(loss)
                if done:
                    if verbose:
                        print("Episode finished after {} timesteps. Total reward: {}.".format(t, total_reward))
                    break
            if not done:
                if verbose:
                    print("Episode timed-out after {} timesteps. Total reward: {}.".format(t, total_reward))
            episode_rewards.append(total_reward)
            if train:
                losses.append(np.mean(episode_losses))
            else:
                losses.append(0)
        if verbose:
            print("Average reward: {}".format(np.mean(episode_rewards)))
        return episode_rewards, losses
