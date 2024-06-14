

import jax.numpy as jnp
from src.bin.referJaxOps.ops_utils import contract_dimensions, broadcast

"""
NB: 
    jnp.dot -> doesn't work for dot_general due to switching of dimensions.
    jnp.select -> doesn't work for select_n operation, wrong result. 
"""


def transpose(operand, permutation):
    return jnp.transpose(operand, permutation)

############################## Arithmetic Operations ############################

def add(x, y):
    return jnp.add(x, y)

def divide(x, y):
    return jnp.divide(x, y)

def max(x, y):
    return jnp.maximum(x, y)

def gt(x, y):
    return jnp.greater(x, y)

########################### Reduction/Expansion Operations ####################

def select_n(which, *cases):
    return jnp.choose(which.astype("int"), cases)

def reduce_sum(operand, axes):
    return jnp.sum(operand, axis=axes)

def broadcast_in_dim(operand, shape, broadcast_dimensions):
    return broadcast(operand, shape, broadcast_dimensions)

def dot_general(lhs, rhs, dimension_numbers, precision = None, preferred_element_type= None):
    return contract_dimensions(lhs, rhs, dimension_numbers)
