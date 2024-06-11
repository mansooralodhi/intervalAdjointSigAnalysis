
import jax
import jax.numpy as jnp
import numpy as np
from src.thesis_prototype_v5.interpreter.ivalOps.intervalArithmetic import IntervalArithmetic
ivalHandler = IntervalArithmetic(jnp)

"""
add operation wrt model jaxpr requirements.
"""


##################################### Transformation Operations ########################
def transpose(x, permutation):
    return ivalHandler.transpose(x, permutation)

def broadcast_in_dim(operand, shape, broadcast_dimensions):
    in_reshape = np.ones(len(shape), dtype=jnp.int32) # NB: if this is changed to jnp then we get trace error __index__
    for i, bd in enumerate(broadcast_dimensions):
        in_reshape[bd] = operand.shape[i]
    return jnp.broadcast_to(jnp.reshape(operand, in_reshape), shape)

###################################### Condition Operations ##############################
def max(x, y):
    return ivalHandler.maximum(x, y)

def min(x, y):
    return ivalHandler.minimum(x, y)

def gt(x, y):
    return ivalHandler.greater_than(x, y)

def eq(x, y):
    return ivalHandler.equal(x, y)

def select_n(which, *cases):
    return ivalHandler.choose(which, *cases)

####################################### Arithmetic Operations ############################
def neg(x):
    return ivalHandler.negative(x)

def add(x, y):
    return ivalHandler.add(x, y)

def sub(x, y):
    return ivalHandler.subtract(x, y)

def mul(x, y):
    return ivalHandler.multiply(x, y)

def divide(x, y):
    return ivalHandler.divide(x, y)

def dot_general(lhs, rhs, dimension_numbers, precision = None, preferred_element_type= None):
    (axes, (_, _)) = dimension_numbers
    return ivalHandler.tensordot(lhs, rhs, axes)

def integer_pow(x, y):
    return ivalHandler.integer_pow(x, y)

##################################### Log/Exp/Power Operations ############################
def tanh(x):
    return ivalHandler.tanh(x)

def exp(x):
    return ivalHandler.exp(x)

def logistic(x):
    return ivalHandler.logistic(x)

##################################### Reduction/Expansion Operations #######################
def reduce_sum(operand, axes):
    return ivalHandler.sum(operand, axis=axes)

def squeeze(operand, dimensions):
    return ivalHandler.squeeze(operand, dimensions)

def slice(operand, start_indices, limit_indices=None, strides=None):
    if isinstance(operand, tuple):
        return jax.lax.slice(operand[0],start_indices, limit_indices, strides), jax.lax.slice(operand[1],start_indices, limit_indices, strides)
    return jax.lax.slice(operand, start_indices, limit_indices, strides)

def pad(operand, pad_width, padding_config):
    if isinstance(operand, tuple):
        return jax.lax.pad(operand[0], pad_width, padding_config), jax.lax.pad(operand[1], pad_width, padding_config)
    return jax.lax.pad(operand, pad_width, padding_config)

########################################### Utils #########################################
def expm1(x):
    return ivalHandler.expm1(x, )

def convert_element_type(x, new_dtype, weak_type):
    if isinstance(x, tuple):
        return jax.lax.convert_element_type(x[0], new_dtype, weak_type), jax.lax.convert_element_type(x[1], new_dtype, weak_type)
    return jax.lax.convert_element_type(x, new_dtype)
