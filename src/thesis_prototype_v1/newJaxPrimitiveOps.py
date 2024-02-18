"""
JAX-traceable function must be able to operate not only on concrete arguments
but also on special abstract arguments that JAX may use to abstract the
function execution.

The JAX traceability property is satisfied as long as the function is written in
terms of JAX primitives.
"""

import jax
import numpy as np
from jax import core
from jax._src import api

"""
Note:       New primitives are defined for jax types and not for pytrees,
            hence, new primitives might work with pytrees and do higher
            order transformations but they will not work with jit.
        
Question:   With new primitive, 
            why is jacobian computed with pytree-intervals input
            but not with simple float input unless all the xla
            implementation is defined ???

Source: https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html
"""

multiply_add_p = core.Primitive("multiply_add")  # Create the primitive


def multiply_add_impl(x, y, z):
    """Primal Evaluation Rule"""
    """Concrete implementation of the primitive.
    
  This function does not need to be JAX traceable.
  Args:
    x, y, z: the concrete arguments of the primitive. Will only be called with
      concrete values.
  Returns:
    the concrete result of the primitive.
  """
    # Note that we can use the original numpy, which is not JAX traceable
    return np.add(np.multiply(x, y), z)


# Now we register the primal implementation with JAX
multiply_add_p.def_impl(multiply_add_impl)


def multiply_add_prim(x, y, z):
    """The JAX-traceable way to use the JAX primitive.

    Note that the traced arguments must be passed as positional arguments
    to `bind`.
    """
    return multiply_add_p.bind(x, y, z)


def square_add_prim(a, b):
    """A square-add function implemented using the new JAX-primitive."""
    return multiply_add_prim(a, a, b)


# def multiply_add_abstract_eval(xs, ys, zs):
#     """Abstract evaluation of the primitive.
#
#   This function does not need to be JAX traceable. It will be invoked with
#   abstractions of the actual arguments.
#   Args:
#     xs, ys, zs: abstractions of the arguments.
#   Result:
#     a ShapedArray for the result of the primitive.
#   """
#     assert xs.shape == ys.shape
#     assert xs.shape == zs.shape
#     return core.ShapedArray(xs.shape, xs.dtype)


# Now we register the abstract evaluation with JAX
# multiply_add_p.def_abstract_eval(multiply_add_abstract_eval)

from intervalArithmetic import IntervalArithmetic

x = IntervalArithmetic(1.0, 2.0)
# x = 2.0
print(square_add_prim(x, x))
print(api.jit(square_add_prim)(x, x))
# print(api.jacfwd(square_add_prim)(x, x)) # only works with pytree-intervals
# print(jax.jacfwd(multiply_add_impl)(x, x, x)) # only works with pytree-intervals

