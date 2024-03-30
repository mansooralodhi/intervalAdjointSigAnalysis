

"""
Input Argument
---------------
1. Float
2. Intervals with shape (2,)
3. Vector of intervals each with shape (2,)
4. Matrix of intervals each with shape (2,)

"""

import jax.numpy as jnp


def div(x, y):
    # fixme, really ???
    #  in case of jnp.mean the value of y is 2 since the interval
    #  len is 2, practically is should not divide by 2
    if y.size == 1 or x.size == 1:
        # constant case or jnp.mean case
        return x / y
    return jnp.stack((x[..., 0] / y[..., 0], x[..., 1] / y[..., 1]), axis=-1)


def mul(x, y):
    # fixme
    if isinstance(x, float) or isinstance(y, float):
        # case in Neils approach
        return x * y
    if y.size == 1 or x.size == 1:
        # constant case
        return y * x
    return jnp.stack((x[..., 0] * y[..., 0], x[..., 1] * y[..., 1]), axis=-1)


def add(x, y):
    # fixme
    return jnp.stack((x[..., 0] + y[..., 0], x[..., 1] + y[..., 1]), axis=-1)

def sub(x, y):
    # fixme
    return jnp.stack((x[..., 0] - y[..., 0], x[..., 1] - y[..., 1]), axis=-1)


def exp(x):
    # fixme
    return jnp.stack((jnp.exp(x[..., 0]), jnp.exp(x[..., 1])), axis=-1)


def reduce_sum(x):
    # fixme
    return jnp.asarray([jnp.sum(x[..., 0]), jnp.sum(x[..., 1])])


def reduce_max(x):
    # fixme
    return jnp.asarray([jnp.max(x[..., 0]), jnp.max(x[..., 1])])
