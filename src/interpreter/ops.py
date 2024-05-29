
from jax import lax
import jax.numpy as jnp
import numpy as np
from src.interpreter.intervalsOps.intervalArithmetic import IntervalArithmetic
ivalHandler = IntervalArithmetic(jnp)

"""
add operation wrt model jaxpr requirements.
"""


def transpose(x, permutation):
    return ivalHandler.transpose(x, permutation)

def tanh(x):
    return ivalHandler.tanh(x)

def logistic(x):
    return ivalHandler.logistic(x)

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

def min(x, y):
    return ivalHandler.minimum(x, y)

def gt(x, y):
    return ivalHandler.greater_than(x, y)

# def convert_element_type(x, new_dtype, weak_type):
#     return ivalHandler.convert_element_type(x, new_dtype, weak_type)

def expm1(x, ):
    return ivalHandler.expm1(x, )

########################### Reduction/Expansion Operations ####################

def select_n(which, *cases):
    return ivalHandler.choose(which, *cases)

def reduce_sum(operand, axes):
    return ivalHandler.sum(operand, axis=axes)


def dot_general(lhs, rhs, dimension_numbers, precision = None, preferred_element_type= None):
    (axes, (_, _)) = dimension_numbers
    return ivalHandler.tensordot(lhs, rhs, axes)

def broadcast_in_dim(operand, shape, broadcast_dimensions):
    in_reshape = np.ones(len(shape), dtype=jnp.int32) # NB: if this is changed to jnp then we get trace error __index__
    for i, bd in enumerate(broadcast_dimensions):
        in_reshape[bd] = operand.shape[i]
    return jnp.broadcast_to(jnp.reshape(operand, in_reshape), shape)

def slice(operand, start_indices, limit_indices, strides=None):
    # todo: verify
    return
    if isinstance(operand, tuple):
        return lax.slice(operand[0], start_indices, limit_indices, strides), lax.slice(operand[1], start_indices, limit_indices, strides)
    return lax.slice(operand, start_indices, limit_indices, strides)

def sqeeze(array, dimensions):
    # todo: verify
    return
    if isinstance(array, tuple):
        return lax.squeeze(array[0], dimensions), lax.squeeze(array[1], dimensions)
    return lax.squeeze(array, dimensions)

def pad(operand, padding_value, padding_config):
    # todo: verify
    return
    if isinstance(operand, tuple):
        return lax.pad(operand[0], padding_value, padding_config), lax.pad(operand[1], padding_value, padding_config)
    return lax.pad(operand, padding_value, padding_config)