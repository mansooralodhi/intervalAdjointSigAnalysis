
# Importing Jax functions useful for tracing/interpreting.

from jax._src.util import safe_map
import numpy as np
from jax import lax
from jax.core import Jaxpr, Literal
from typing import Union, Tuple, List

interval_registry = dict()
interval_registry[lax.tanh_p] = np.tanh
interval_registry[lax.exp_p] = np.exp
interval_registry[lax.add_p] = np.add
interval_registry[lax.mul_p] = np.multiply
interval_registry[lax.dot_general_p] = np.matmul



class EvalJaxpr(object):

    env = {}

    @classmethod
    def read(cls, var):
        # Literals are values baked into the Jaxpr
        if type(var) is Literal:
            return var.val
        # print(f"Reading from env: var= {var}, val= {cls.env[var]}")
        return cls.env[var]

    @classmethod
    def write(cls, var, val):
        # print(f"Writing in env: var= {var}, val= {val}")
        cls.env[var] = val

    @classmethod
    def eval_jaxpr(cls, jaxpr: Jaxpr, consts: Union[Tuple, List], *args: Union[Tuple, List]):
        # Mapping from variable -> value

        # Bind args and consts to environment
        safe_map(cls.write, jaxpr.invars, args)
        safe_map(cls.write, jaxpr.constvars, consts)

        # Loop through equations and evaluate primitives using `bind`
        for eqn in jaxpr.eqns:
            # Read inputs to equation from environment
            invals = safe_map(cls.read, eqn.invars)
            if eqn.primitive not in interval_registry:
                raise NotImplementedError(
                    f"{eqn.primitive} does not have numpy implementation.")
            new_primitive = interval_registry[eqn.primitive]
            outvals = new_primitive(*invals)
            """ Assuming a unary function """
            # Write the results of the primitive into the environment
            safe_map(cls.write, eqn.outvars, [outvals])
            # Read the final result of the Jaxpr from the environment
        output = safe_map(cls.read, jaxpr.outvars)
        cls.env = {}
        return output









