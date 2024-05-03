

import jax.numpy as jnp
from src.interpreter.intervalsOps.intervalArithmetic import IntervalArithmetic
ivalHandler = IntervalArithmetic(jnp)

"""
add operation wrt model jaxpr requirements.
"""


def transpose(x, permutation):
    return ivalHandler.transpose(x, permutation)

def tanh(x):
    return ivalHandler.tanh(x)

############################## Arithmetic Operations ############################

def add(x, y):
    return ivalHandler.add(x, y)

def sub(x, y):
    return ivalHandler.subtract(x, y)

def mul(x, y):
    return ivalHandler.multiply(x, y)

def divide(x, y):
    return ivalHandler.divide(x, y)

def max(x, y):
    return ivalHandler.maximum(x, y)

def gt(x, y):
    return ivalHandler.greater_than(x, y)

########################### Reduction/Expansion Operations ####################

def select_n(which, *cases):
    return ivalHandler.choose(which, *cases)

def reduce_sum(operand, axes):
    return ivalHandler.sum(operand, axis=axes)

def dot_general(lhs, rhs, dimension_numbers, precision = None, preferred_element_type= None):
    (axes, (_, _)) = dimension_numbers
    return ivalHandler.tensordot(lhs, rhs, axes)

def broadcast_in_dim(operand, shape, broadcast_dimensions):
    in_reshape = jnp.ones(len(shape), dtype=jnp.int32)
    for i, bd in enumerate(broadcast_dimensions):
        in_reshape[bd] = operand.shape[i]
    return jnp.broadcast_to(jnp.reshape(operand, in_reshape), shape)


