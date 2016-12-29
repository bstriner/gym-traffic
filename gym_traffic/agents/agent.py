from abc import ABCMeta, abstractmethod
from gym.utils import seeding


class Agent(object):
    __metaclass__ = ABCMeta

    def __init__(self, input_space, action_space, seed=None):
        self.input_space = input_space
        self.action_space = action_space
        self._seed(seed=seed)

    @abstractmethod
    def new_episode(self):
        pass

    @abstractmethod
    def observe(self, observation):
        ''' To override '''
        pass

    @abstractmethod
    def act(self):
        ''' To override '''
        pass

    @abstractmethod
    def learn(self, action, reward, done):
        ''' To override '''
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
