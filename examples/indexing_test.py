import os

os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"

import keras.backend as K
import numpy as np

n = 5
m = 3

x = np.random.random((n, m))
a = np.random.randint(0, m, (n, 1))
y = x[range(n), a.reshape((-1,))]
print x
print a
print y

_x = K.variable(x)
_a = K.variable(a, dtype="int32")
_y = K.reshape(_x[K.arange(K.shape(_x)[0]), K.reshape(_a, (-1,))], (-1,1))
f = K.function([],[_y])
print "Test1"
print f([])

_y = K.sum(K.one_hot(K.reshape(_a,(-1,)), K.shape(_x)[1]) * _x, axis=-1, keepdims=True)
f = K.function([],[_y])
print "Test2"
print f([])
