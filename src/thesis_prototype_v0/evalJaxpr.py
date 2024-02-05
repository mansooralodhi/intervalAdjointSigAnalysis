from intervalArithmetic import  IntervalArithmetic
# Importing Jax functions useful for tracing/interpreting.

import numpy as np
from functools import wraps
import jax
import jax.numpy as jnp
from jax import core
from jax import lax
from jax._src.util import safe_map

inverse_registry = {}
inverse_registry[lax.sin_p] = np.sin

def f(x):
    return jnp.sin(jnp.sin(x))

# inverse_registry[lax.exp_p] = jnp.log
# inverse_registry[lax.tanh_p] = jnp.arctanh

# def f(x):
#   return jnp.exp(jnp.tanh(x))

def eval_jaxpr(jaxpr, consts, *args):
    # Mapping from variable -> value
    env = {}

    def read(var):
        # Literals are values baked into the Jaxpr
        if type(var) is core.Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    # Bind args and consts to environment
    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    # Loop through equations and evaluate primitives using `bind`
    for eqn in jaxpr.eqns:
        # Read inputs to equation from environment
        invals = safe_map(read, eqn.invars)
        # `bind` is how a primitive is called
        outvals = eqn.primitive.bind(*invals, **eqn.params)
        # Primitives may return multiple outputs or not
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        # Write the results of the primitive into the environment
        safe_map(write, eqn.outvars, outvals)
        # Read the final result of the Jaxpr from the environment
    return safe_map(read, jaxpr.outvars)


def inverse_jaxpr(jaxpr, consts, *args):
    env = {}

    def read(var):
        if type(var) is core.Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    # Args now correspond to Jaxpr outvars
    safe_map(write, jaxpr.outvars, args)
    safe_map(write, jaxpr.constvars, consts)

    # Looping backward
    for eqn in jaxpr.eqns[::-1]:
        #  outvars are now invars
        invals = safe_map(read, eqn.outvars)
        if eqn.primitive not in inverse_registry:
            raise NotImplementedError(
                f"{eqn.primitive} does not have registered inverse.")
        # Assuming a unary function
        x = IntervalArithmetic(1.0, 2.0)
        outval = inverse_registry[eqn.primitive](x)
        # outval = inverse_registry[eqn.primitive](*invals)
        safe_map(write, eqn.invars, [outval])
    return safe_map(read, jaxpr.invars)


def inverse(fun):
  @wraps(fun)
  def wrapped(*args, **kwargs):
    # Since we assume unary functions, we won't worry about flattening and
    # unflattening arguments.
    closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
    out = inverse_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
    return out[0]
  return wrapped



f_inv = inverse(f)
print(f(1.0))
print(f_inv(1.0))
