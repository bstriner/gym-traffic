from agent import Agent


class EpsilonExplorer(Agent):
    def __init__(self, agent, epsilon=0.1, decay=1e-6, seed=None):
        self.agent = agent
        self.step = 0
        self.epsilon = epsilon
        self.decay = decay
        super(EpsilonExplorer, self).__init__(agent.input_space, agent.action_space, seed=seed)

    def __getattr__(self, item):
        return getattr(self.agent, item)

    def new_episode(self):
        return self.agent.new_episode()

    def observe(self, observation):
        return self.agent.observe(observation)

    def act(self):
        self.step += 1
        epsilon = self.epsilon * (1.0 / (1.0 + self.step * self.decay))
        if self.np_random.uniform(0, 1) > epsilon:
            return self.agent.act()
        else:
            return self.action_space.sample()

    def learn(self, action, reward, done):
        return self.agent.learn(action, reward, done)
