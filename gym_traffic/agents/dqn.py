from keras.models import Model
from keras.layers import Dense, Flatten, LeakyReLU, Input, merge, Reshape, Lambda
from keras.regularizers import L1L2Regularizer
from keras.utils.np_utils import to_categorical
import numpy as np
from gym import spaces
from keras.optimizers import Adam
import itertools
from keras import backend as K
import os
from gym.utils import seeding


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


class DQN(object):
    def __init__(self, input_space, output_space, memory_size=20, replay_size=50, decay=0.9, epsilon=0.05, seed=None):
        self._seed(seed=seed)
        self.input_space = input_space
        self.input_dim = calc_input_dim(input_space)
        self.output_space = output_space
        self.memory_size = memory_size
        self.replay_size = replay_size
        self.decay = K.variable(K.cast_to_floatx(decay))
        self.epsilon = epsilon
        self.data_dim = self.input_dim * self.memory_size
        self.replay = []
        self.new_episode()
        if not isinstance(output_space, spaces.Discrete):
            raise NotImplementedError("Only Discrete output spaces supported")
        self.build_network()

    def new_episode(self):
        self.memory = [np.zeros((1, self.input_dim)) for i in range(self.memory_size)]
        self.observation = None
        self.last_observation = None

    def build_network(self):
        hidden_dim = 512
        reg = lambda: L1L2Regularizer(l1=1e-5)
        x = Input(shape=(self.data_dim,), name="x")
        h = x
        h = Dense(hidden_dim, W_regularizer=reg())(h)
        h = LeakyReLU(0.2)(h)
        h = Dense(hidden_dim, W_regularizer=reg())(h)
        h = LeakyReLU(0.2)(h)
        h = Dense(hidden_dim, W_regularizer=reg())(h)
        h = LeakyReLU(0.2)(h)
        y = Dense(self.output_space.n)(h)
        # Q(s, a)
        self.Q = Model(x, y, name="Q")

        action = Input(shape=(1,), dtype='int32', name="action")
        selected_y = merge([y, action], mode=lambda z: z[0][:, z[1]], output_shape=lambda z: z[1])
        self.Q_s = Model([x, action], selected_y, name="Q_s")

        value = Lambda(lambda z: K.max(z, axis=1, keepdims=True), output_shape=lambda z: (z[0], 1))(y)
        self.V = Model(x, value, name="V")

        x_prime = Input(shape=(self.data_dim,), name="x_prime")
        v_prime = Lambda(lambda z: K.stop_gradient(z), output_shape=lambda z: z)(self.V(x_prime))
        q = self.Q_s([x, action])

        r_pred = merge([q, v_prime], mode=lambda z: z[0] - self.decay * z[1], output_shape=lambda z: z[0])

        self.training_model = Model([x, action, x_prime], r_pred, name="training_model")
        opt = Adam(1e-4)
        self.training_model.compile(opt, "mean_squared_error")

    def observe(self, observation):
        observation = concat_input(observation, self.input_space)
        self.memory = self.memory[1:] + [observation]
        self.last_observation = self.observation
        self.observation = np.hstack(self.memory)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def act(self):
        if self.np_random.uniform(0,1) >= self.epsilon:
            preds = self.Q.predict(self.observation.reshape((1, -1))).reshape((-1,))
            action = np.argmax(preds)
            return action
        else:
            return self.output_space.sample()

    def combined_replay(self):
        return [np.vstack(x[i] for x in self.replay) for i in range(4)]

    def learn(self, action, reward):
        datum = [self.last_observation, action, self.observation, reward]
        self.replay.append(datum)
        if len(self.replay) > self.replay_size:
            self.replay.pop(0)

        data = self.combined_replay()
        self.training_model.train_on_batch(data[0:3], data[3])

    def save(self, filepath):
        dirpath = os.path.dirname(filepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.Q.save_weights(filepath)

    def load(self, filepath):
        self.Q.load_weights(filepath)
