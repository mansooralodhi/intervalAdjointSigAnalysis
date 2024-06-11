
# Importing Jax functions useful for tracing/interpreting.

from jax._src.util import safe_map

from jax.core import Jaxpr, Literal
from typing import Tuple, List, Union

def eval_jaxpr(cls, jaxpr: Jaxpr, consts: Union[Tuple, List], *args: Union[Tuple, List]):
    # Mapping from variable -> value
    env = {}

    def read(var):
        # Literals are values baked into the Jaxpr
        if type(var) is Literal:
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
        if eqn.primitive not in interval_registry:
            raise NotImplementedError(f"{eqn.primitive} does not have numpy implementation.")
        new_primitive = interval_registry[eqn.primitive]
        outvals = new_primitive(*invals)
        """ Assuming a unary function """
        # Write the results of the primitive into the environment
        safe_map(write, eqn.outvars, [outvals])
        # Read the final result of the Jaxpr from the environment
    output = safe_map(read, jaxpr.outvars)
    return output









