from keras.models import Model
from keras.layers import Dense, Flatten, LeakyReLU, Input, merge, Reshape, Lambda, BatchNormalization, Dropout
from keras.regularizers import L1L2Regularizer
from keras.utils.np_utils import to_categorical
import numpy as np
from gym import spaces
from keras.optimizers import Adam
import itertools
from keras import backend as K
import os
from agent import Agent
from random import shuffle


def flatten_spaces(space):
    if isinstance(space, spaces.Tuple):
        return list(itertools.chain.from_iterable(flatten_spaces(s) for s in space.spaces))
    else:
        return [space]


def calc_input_dim(space):
    dims = []
    print "Space: {}".format(space)
    print "Flattened: {}".format(flatten_spaces(space))
    for i in flatten_spaces(space):
        if isinstance(i, spaces.Discrete):
            dims.append(i.n)
        elif isinstance(i, spaces.Box):
            dims.append(np.prod(i.shape))
        else:
            raise NotImplementedError("Only Discrete and Box input spaces currently supported")
    return np.sum(dims)


def concat_input(observation, input_space):
    if isinstance(input_space, spaces.Tuple):
        return np.hstack([np.array(concat_input(obs, space)) for obs, space in
                          zip(observation, input_space.spaces)])
    elif isinstance(input_space, spaces.Discrete):
        return to_categorical(observation, nb_classes=input_space.n).reshape((1, -1))
    elif isinstance(input_space, spaces.Box):
        return observation.reshape((1, -1))
    else:
        raise NotImplementedError("Only Discrete and Box input spaces currently supported")


class DQN(Agent):
    def __init__(self, input_space, action_space, memory_size=10, replay_size=64, discount=0.95, seed=None,
                 optimizer=None):
        super(DQN, self).__init__(input_space, action_space, seed=seed)
        self.input_dim = calc_input_dim(input_space)
        self.memory_size = memory_size
        self.replay_size = replay_size
        self.discount = K.variable(K.cast_to_floatx(discount))
        self.step = 0
        self.data_dim = self.input_dim * self.memory_size
        self.replay = []
        self.new_episode()
        if optimizer is None:
            optimizer = Adam(1e-4, decay=1e-6)
        self.optimizer = optimizer
        if not isinstance(action_space, spaces.Discrete):
            raise NotImplementedError("Only Discrete action spaces supported")
        self.build_network()

    def new_episode(self):
        self.memory = [np.zeros((1, self.input_dim)) for i in range(self.memory_size)]
        self.observation = None
        self.last_observation = None

    def build_network(self):
        hidden_dim = 1024
        reg = lambda: L1L2Regularizer(l1=1e-9, l2=1e-9)
        x = Input(shape=(self.data_dim,), name="x")
        h = x
        h = Dense(hidden_dim, W_regularizer=reg())(h)
        h = Dropout(0.5)(h)
        #h = BatchNormalization(mode=1)(h)
        h = LeakyReLU(0.2)(h)
        h = Dense(hidden_dim / 2, W_regularizer=reg())(h)
        h = Dropout(0.5)(h)
        #h = BatchNormalization(mode=1)(h)
        h = LeakyReLU(0.2)(h)
        h = Dense(hidden_dim / 4, W_regularizer=reg())(h)
        h = Dropout(0.5)(h)
        #h = BatchNormalization(mode=1)(h)
        h = LeakyReLU(0.2)(h)
        y = Dense(self.action_space.n, W_regularizer=reg())(h)
        # Q(s, a)
        self.Q = Model(x, y, name="Q")

        action = Input(shape=(1,), dtype='int32', name="action")
        """
        selected_y = merge([y, action],
                           mode=lambda z: K.sum(K.one_hot(K.reshape(z[1], (-1,)), K.shape(z[0])[1]) * z[0], axis=-1,
                                                keepdims=True), output_shape=lambda z: z[1])
                                                """
        selected_y = merge([y, action],
                           mode=lambda z: K.reshape(z[0][K.arange(K.shape(z[0])[0]), K.reshape(z[1], (-1,))], (-1, 1)),
                           output_shape=lambda z: z[1])

        self.Q_s = Model([x, action], selected_y, name="Q_s")

        value = Lambda(lambda z: K.max(z, axis=-1, keepdims=True), output_shape=lambda z: (z[0], 1))(y)
        self.V = Model(x, value, name="V")

        x_prime = Input(shape=(self.data_dim,), name="x_prime")
        done = Input(shape=(1,), name="done", dtype="int32")
        v_prime = Lambda(lambda z: K.stop_gradient(z), output_shape=lambda z: z)(self.V(x_prime))
        # v_prime = self.V(x_prime)
        q = self.Q_s([x, action])

        r_pred = merge([q, v_prime, done], mode=lambda z: z[0] - ((1 - z[2]) * self.discount * z[1]),
                       output_shape=lambda z: z[0])

        self.training_model = Model([x, action, x_prime, done], r_pred, name="training_model")

        self.training_model.compile(self.optimizer, "mean_squared_error")

    def observe(self, observation):
        observation = concat_input(observation, self.input_space)
        self.memory = self.memory[1:] + [observation]
        self.last_observation = self.observation
        self.observation = np.hstack(self.memory)

    def act(self):
        preds = self.Q.predict(self.observation.reshape((1, -1))).reshape((-1,))
        action = np.argmax(preds)
        return action

    def combined_replay(self):
        return [np.vstack(x[i] for x in self.replay) for i in range(5)]

    def learn(self, action, reward, done):
        datum = [self.last_observation, action, self.observation, [[1]] if done else [[0]], reward]
        self.replay.append(datum)
        if len(self.replay) > self.replay_size:
            self.replay.pop(0)
        # shuffle(self.replay)
        data = self.combined_replay()
        loss = self.training_model.train_on_batch(data[0:4], data[4])

        return loss

    def save(self, filepath):
        dirpath = os.path.dirname(filepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.Q.save_weights(filepath)

    def load(self, filepath):
        self.Q.load_weights(filepath)
