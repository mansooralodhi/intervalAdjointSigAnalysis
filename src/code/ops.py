import numpy as np
import jax.numpy as jnp

"""
Input Array Types
-----------------
1.  Vector
2.  Matrix

"""


def div(x, y):
    # fixme, really ???
    #  in case of jnp.mean the value of y is 2 since the interval
    #  len is 2, practically is should not divide by 2
    if y.size == 1 or x.size == 1:
        # constant case or jnp.mean case
        return x / y
    return jnp.stack((x[..., 0] / y[..., 0], x[..., 1] / y[..., 1]), axis=-1)


def mul(x, y):
    if isinstance(x, float) or isinstance(y, float):
        # case in Neils approach
        return x * y
    if y.size == 1 or x.size == 1:
        # constant case
        return y * x
    return jnp.stack((x[..., 0] * y[..., 0], x[..., 1] * y[..., 1]), axis=-1)


def add(x, y):
    return jnp.stack((x[..., 0] + y[..., 0], x[..., 1] + y[..., 1]), axis=-1)


def sub(x, y):
    return jnp.stack((x[..., 0] - y[..., 0], x[..., 1] - y[..., 1]), axis=-1)


def exp(x):
    return jnp.stack((jnp.exp(x[..., 0]), jnp.exp(x[..., 1])), axis=-1)


def reduce_sum(x):
    return jnp.asarray([jnp.sum(x[..., 0]), jnp.sum(x[..., 1])])


def reduce_max(x):
    return jnp.asarray([jnp.max(x[..., 0]), jnp.max(x[..., 1])])
