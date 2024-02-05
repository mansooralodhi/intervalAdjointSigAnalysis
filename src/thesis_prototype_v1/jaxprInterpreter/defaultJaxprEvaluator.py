
# Importing Jax functions useful for tracing/interpreting.

import jax
import jax.numpy as jnp
from jax import core
from jax._src.util import safe_map


class DefaultJaxprEvaluator(object):


    env = {}

    @classmethod
    def read(cls, var):
        # Literals are values baked into the Jaxpr
        if type(var) is core.Literal:
            return var.val
        # print(f"Reading from env: var= {var}, val= {cls.env[var]}")
        return cls.env[var]

    @classmethod
    def write(cls, var, val):
        # print(f"Writing in env: var= {var}, val= {val}")
        cls.env[var] = val

    @classmethod
    def eval_jaxpr(cls, jaxpr, consts, *args):
        # Mapping from variable -> value

        # Bind args and consts to environment
        safe_map(cls.write, jaxpr.invars, args)
        safe_map(cls.write, jaxpr.constvars, consts)

        # Loop through equations and evaluate primitives using `bind`
        for eqn in jaxpr.eqns:
            # Read inputs to equation from environment
            invals = safe_map(cls.read, eqn.invars)
            # `bind` is how a primitive is called
            outvals = eqn.primitive.bind(*invals, **eqn.params)
            # Primitives may return multiple outputs or not
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            """ Assuming a unary function """
            # Write the results of the primitive into the environment
            safe_map(cls.write, eqn.outvars, outvals)
            # Read the final result of the Jaxpr from the environment
        output = safe_map(cls.read, jaxpr.outvars)
        cls.env = {}
        return output


if __name__ == "__main__":

    def f(x):
        return jnp.exp(jnp.tanh(x))

    closed_jaxpr = jax.make_jaxpr(f)(jnp.ones(5))
    otuput = DefaultJaxprEvaluator.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, jnp.ones(5))
    print(otuput)














