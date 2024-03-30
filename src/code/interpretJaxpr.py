"""
"""
"""
Input: float | jnp.ndarray

NB: Jaxpr interpret should know if the jnp.ndarray is derived from np.ndarray(dtype=interval).

    If the jnp.ndarray is derived 
        than execute custom jax primitive ops
    else
        execute jax primitive ops   
        
2.  interpret_jaxpr
    2.1 preliminaries:
        2.1.1 write jax primitive ops for interval inputs
        2.1.2 create registry
        2.1.3 store primitive ops in registry
    2.2 create env
    2.3 store variable values in env
    2.4 store constants values in env
    2.5 find eqn in jaxpr
        2.5.1 get values from env
        2.5.2 get method from registry 
        2.5.3 get output 
        2.5.4 store output in env
    2.6 get output from env
    2.7 return output
"""

from jax import core
from jax._src.util import safe_map
from src.code.setupRegistry import get_registry

registry = get_registry()


def interpret_jaxpr(jaxpr, consts, *args):
    env = {}

    def read(var):
        if type(var) is core.Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    for eqn in jaxpr.eqns:

        invals = safe_map(read, eqn.invars)

        if eqn.primitive not in registry:
            raise NotImplementedError(f"{eqn.primitive} does not have registered interval equivalent.")

        outvals = registry[eqn.primitive](*invals, **eqn.params)

        # Primitives may return multiple outputs or not
        # if eqn.primitive.multiple_results:
        outvals = [outvals]

        safe_map(write, eqn.outvars, outvals)

    return safe_map(read, jaxpr.outvars)
