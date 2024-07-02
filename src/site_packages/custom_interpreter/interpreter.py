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

import jax
from jax._src.util import safe_map
from jax.experimental.pjit import pjit_p
from src.site_packages.custom_interpreter.registry import registry
from jax.custom_derivatives import custom_vjp_call_p, custom_vjp_call_jaxpr_p, custom_jvp_call_p

from copy import deepcopy
from typing import Union, Literal
from src.site_packages.interval_arithmetic.numpyLike import Interval, NDArray


def safe_interpreter(jaxpr: jax.make_jaxpr, literals: list, *args: Union[Interval, NDArray]) -> object:
    """
    jaxpr attributes:
        constvars | invars | eqns (iterated) | outvars (ignored)
    consts:
        list of static params/weights
    args:
        list[...]
        len(args) == len(jaxpr.invars)
    """
    env = {}

    def write(vars, args):
        env[vars] = args

    def read(vars):
        val = vars.val if type(vars) is jax.core.Literal else env[vars]
        return val

    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, literals)

    for eqn in jaxpr.eqns:

        inVarValues = safe_map(read, eqn.invars)

        if eqn.primitive in [custom_vjp_call_p, custom_jvp_call_p]:
            sub_closedJaxpr = eqn.params['call_jaxpr']
            outVarValues = safe_interpreter(sub_closedJaxpr.jaxpr, sub_closedJaxpr.literals, *inVarValues)
            outVarValues = outVarValues if isinstance(outVarValues, list | tuple) else [outVarValues]

        elif eqn.primitive == custom_vjp_call_jaxpr_p:
            sub_closedJaxpr = eqn.params['fun_jaxpr']
            outVarValues = safe_interpreter(sub_closedJaxpr.jaxpr, sub_closedJaxpr.literals, *inVarValues)
            outVarValues = outVarValues if isinstance(outVarValues, list | tuple) else [outVarValues]

        elif eqn.primitive == pjit_p:
            sub_closedJaxpr = eqn.params['jaxpr']
            outVarValues = safe_interpreter(sub_closedJaxpr.jaxpr, sub_closedJaxpr.literals, *inVarValues)
            outVarValues = outVarValues if isinstance(outVarValues, list | tuple) else [outVarValues]

        elif eqn.primitive in registry:
            outVarValues = [registry[eqn.primitive](*inVarValues, **eqn.params)]

        else:
            raise NotImplementedError(f"{eqn.primitive} does not have registered interval equivalent.")

        safe_map(write, eqn.outvars, outVarValues)

    output = safe_map(read, jaxpr.outvars)

    return output


def _safe_interpreter(jaxpr: jax.make_jaxpr, literals: list, *args: Union[Interval, NDArray]) -> object:
    # fixme:    the current method tracks intermediate primals output wrt to all args,
    #           integrate it with api using a flat.

    """
    jaxpr attributes:
        constvars | invars | eqns (iterated) | outvars (ignored)
    consts:
        list of static params/weights
    args:
        list[...]
        len(args) == len(jaxpr.invars)
    """
    env = {}

    def write(vars, args):
        env[vars] = args

    def read(vars):
        val = vars.val if type(vars) is jax.core.Literal else env[vars]
        return val

    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, literals)

    env_custom = {}
    intermediate_outputs = []

    def write_custom(vars, args):
        env_custom[vars] = args


    # iteratively calls write function for each element of len(const)
    safe_map(write_custom, jaxpr.invars, args)  # store the initial input vector, X
    safe_map(write_custom, jaxpr.constvars, literals)
    intermediate_outputs.append(args[0])

    for eqn in jaxpr.eqns:

        inVarValues = safe_map(read, eqn.invars)

        if eqn.primitive in [custom_vjp_call_p, custom_jvp_call_p]:
            sub_closedJaxpr = eqn.params['call_jaxpr']
            outVarValues = safe_interpreter(sub_closedJaxpr.jaxpr, sub_closedJaxpr.literals, *inVarValues)
            outVarValues = outVarValues if isinstance(outVarValues, list | tuple) else [outVarValues]

        elif eqn.primitive == custom_vjp_call_jaxpr_p:
            sub_closedJaxpr = eqn.params['fun_jaxpr']
            outVarValues = safe_interpreter(sub_closedJaxpr.jaxpr, sub_closedJaxpr.literals, *inVarValues)
            outVarValues = outVarValues if isinstance(outVarValues, list | tuple) else [outVarValues]

        elif eqn.primitive == pjit_p:
            sub_closedJaxpr = eqn.params['jaxpr']
            outVarValues = safe_interpreter(sub_closedJaxpr.jaxpr, sub_closedJaxpr.literals, *inVarValues)
            outVarValues = outVarValues if isinstance(outVarValues, list | tuple) else [outVarValues]

        elif eqn.primitive in registry:
            outVarValues = [registry[eqn.primitive](*inVarValues, **eqn.params)]

        else:
            raise NotImplementedError(f"{eqn.primitive} does not have registered interval equivalent.")

        safe_map(write, eqn.outvars, outVarValues)
        # todo: if inVarValues in env_custom then add the output in intermediate_outputs
        for argId in env_custom.keys():
            if argId in eqn.invars:
                intermediate_outputs.append(*outVarValues)
                break

    output = safe_map(read, jaxpr.outvars)

    return output, intermediate_outputs



if __name__ == "__main__":
    safe_interpreter()
