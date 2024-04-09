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
from jax import core
from jax.custom_derivatives import custom_jvp_call_p
from jax.experimental.pjit import pjit_p
from src.customInterpreter.registry import ops_mapping

registry = ops_mapping()



def forward_interpret(jaxpr: jax.make_jaxpr, consts: list, *args: tuple) -> object:
    """
    jaxpr attributes:
        constvars | invars | eqns (iterated) | outvars (ignored)
    consts:
        list of static params/weights
    args:
        tuple[...]
        len(args) == len(jaxpr.invars)
    """
    env = {}

    # write: input variable and their values in env
    for var, val in zip(jaxpr.invars, args):
        env[var] = val

    # write: input constants and their values in env
    for var, val in zip(jaxpr.constvars, consts):
        env[var] = val

    for eqn in jaxpr.eqns:

        # read: input variable values from env
        inVars = eqn.invars
        inVars = inVars if isinstance(inVars, list) else [inVars]
        inVarValues = list()
        for var in inVars:
            val = var.val if type(var) is core.Literal else env[var]
            inVarValues.append(val)

        if eqn.primitive == custom_jvp_call_p:
            sub_closedJaxpr = eqn.params['call_jaxpr']
            outVarValues = forward_interpret(sub_closedJaxpr.jaxpr, sub_closedJaxpr.literals, *inVarValues)
            # write output variable values in env
            for var, val in zip(eqn.outvars, outVarValues):
                env[var] = val
            continue

        elif eqn.primitive == pjit_p:
            sub_closedJaxpr = eqn.params['jaxpr']
            outVarValues = forward_interpret(sub_closedJaxpr.jaxpr, sub_closedJaxpr.literals, *inVarValues)
            return outVarValues

        elif eqn.primitive in registry:
            outVarValues = registry[eqn.primitive](*inVarValues, **eqn.params)
            outVarValues = [outVarValues]

            # write output variable values in env
            for var, val in zip(eqn.outvars, outVarValues):
                env[var] = val

        else:
            raise NotImplementedError(f"{eqn.primitive} does not have registered interval equivalent.")

    # read: output variable values from env.
    output = list()
    for var in jaxpr.outvars:
        val = env[var]
        output.append(val)

    return output


################################## None-Recursive Interpreter in OOP ########################
# class Interpreter(object):
#
#     def __init__(self):
#         self.env = dict()
#         self.registry = ops_mapping()
#
#     def reset(self):
#         self.env = dict()
#
#     def read(self, var):
#         if type(var) is core.Literal:
#             return var.val
#         return self.env[var]
#
#     def write(self, var, val):
#         self.env[var] = val
#
#     def interpret(self, jaxpr, consts, *args):
#
#         safe_map(self.write, jaxpr.invars, args)
#         safe_map(self.write, jaxpr.constvars, consts)
#
#         for eqn in jaxpr.eqns:
#
#             invals = safe_map(self.read, eqn.invars)
#
#             if eqn.primitive not in self.registry:
#                 raise NotImplementedError(f"{eqn.primitive} does not have registered interval equivalent.")
#
#             outvals = self.registry[eqn.primitive](*invals, **eqn.params)
#
#             # Primitives may return multiple outputs or not
#             # if eqn.primitive.multiple_results:
#             outvals = [outvals]
#
#             safe_map(self.write, eqn.outvars, outvals)
#
#         output = safe_map(self.read, jaxpr.outvars)
#
#         self.reset()
#
#         return output
