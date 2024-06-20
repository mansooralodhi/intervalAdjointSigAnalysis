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
from jax._src.util import safe_map
from jax.experimental.pjit import pjit_p
from src.interpreter.registry import registry
from jax.custom_derivatives import custom_vjp_call_p, custom_vjp_call_jaxpr_p



class Interpreter(object):

    def __init__(self):
        self.registry = registry()

    def safe_interpret(self, jaxpr: jax.make_jaxpr, consts: list, args: list | tuple) -> object:
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
            val = vars.val if type(vars) is core.Literal else env[vars]
            return val

        # iteratively calls write function for each element of len(const)
        safe_map(write, jaxpr.invars, args)
        safe_map(write, jaxpr.constvars, consts)

        for eqn in jaxpr.eqns:

            inVarValues = safe_map(read, eqn.invars)

            if eqn.primitive == custom_vjp_call_p:
                sub_closedJaxpr = eqn.params['call_jaxpr']
                outVarValues = self.safe_interpret(sub_closedJaxpr.jaxpr, sub_closedJaxpr.literals, *inVarValues)
                outVarValues = outVarValues if isinstance(outVarValues, list|tuple) else [outVarValues]

            elif eqn.primitive == custom_vjp_call_jaxpr_p:
                sub_closedJaxpr = eqn.params['fun_jaxpr']
                outVarValues = self.safe_interpret(sub_closedJaxpr.jaxpr, sub_closedJaxpr.literals, inVarValues)
                outVarValues = outVarValues if isinstance(outVarValues, list|tuple) else [outVarValues]

            elif eqn.primitive == pjit_p:
                sub_closedJaxpr = eqn.params['jaxpr']
                outVarValues = self.safe_interpret(sub_closedJaxpr.jaxpr, sub_closedJaxpr.literals, inVarValues)
                outVarValues = outVarValues if isinstance(outVarValues, list|tuple) else [outVarValues]

            elif eqn.primitive in self.registry:
                from jax import lax
                if eqn.primitive == lax.pad_p:
                    print()
                outVarValues = [self.registry[eqn.primitive](*inVarValues, **eqn.params)]

            else:
                print(eqn.primitive)
                raise NotImplementedError(f"{eqn.primitive} does not have registered interval equivalent.")

            safe_map(write, eqn.outvars, outVarValues)

        output = safe_map(read, jaxpr.outvars)
        return output



if __name__ == "__main__":
    Interpreter()