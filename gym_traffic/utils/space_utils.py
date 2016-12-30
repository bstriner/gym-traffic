from gym import spaces
import itertools

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

