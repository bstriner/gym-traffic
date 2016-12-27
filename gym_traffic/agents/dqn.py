from keras.models import Model
from keras.layers import Dense, Flatten, LeakyReLU
from keras.regularizers import L1L2Regularizer


class DQN(object):
    def __init__(self, input_shape, memory_size=20, replay_size=50):
        self.input_shape = input_shape
        self.memory_size = memory_size
        self.replay_size = replay_size

    def step(self, observation, reward):
        return 0
