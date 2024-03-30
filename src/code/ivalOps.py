

"""
Input Arguments
-----------------
1. Float
2. Intervals with shape (2,)
"""

import numpy as np
import jax.numpy as jnp
from typing import Union

_argTypes = Union[float, jnp.ndarray, np.ndarray]


def _verifyBoundaryCondition(ival: Union[jnp.ndarray, np.ndarray]):
    if ival.shape != (2,):
        raise Exception("Incorrect interval shape !")
    # if ival[0] <= ival[1]:
    #     return ival
    return jnp.asarray([ival[1], ival[0]])

def neg(x):
    if isinstance(x, float):
        return -1 * x
    return _verifyBoundaryCondition(-1* x)

def add(x: _argTypes, y: _argTypes):
    # x, y both cannot be float dtypes
    if isinstance(x, float) or isinstance(y, float):
        return x + y
    val = jnp.asarray([x[0] + y[0], x[1] + y[1]])
    return _verifyBoundaryCondition(val)

def sub(x: _argTypes, y: _argTypes):
    # x, y both cannot be float dtypes
    y = -1.0 * y
    y = _verifyBoundaryCondition(y)
    return add(x, y)


def mul(x: _argTypes, y: _argTypes):
    # x, y both cannot be float dtypes
    if isinstance(x, float) and isinstance(y, float):
        return x * y
    if isinstance(x, float) or isinstance(y, float):
        val = jnp.asarray(x * y)
        return _verifyBoundaryCondition(val)
    # now x, y both are np.ndarray or jax.ndarray
    if x.size == 1 or y.size == 1 or (x.size ==1 and y.size==1):
        return _verifyBoundaryCondition(x * y)
    allOutcomes = [x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1]]
    lowerBound = min(allOutcomes)
    upperBound = max(allOutcomes)
    return jnp.asarray([lowerBound, upperBound])

def integer_pow(x, y=None):
    # y is expected to be an integer
    val = jnp.asarray([x[0] ** y, x[1] ** y])
    return _verifyBoundaryCondition(val)

def div(x: _argTypes, y: _argTypes):
    if isinstance(x, float) or isinstance(y, float):
        val = jnp.asarray(x / y)
        return _verifyBoundaryCondition(val)
    y = 1.0/y
    y = _verifyBoundaryCondition(y)
    return mul(x, y)

