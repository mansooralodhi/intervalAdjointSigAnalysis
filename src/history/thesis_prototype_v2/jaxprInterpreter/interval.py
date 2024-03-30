
# Importing Jax functions useful for tracing/interpreting.

import jax
import jax.numpy as jnp
from jax import core
from jax._src.util import safe_map
from registry import interval_registry


class IntervalJaxprEvaluator(object):

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


if __name__ == "__main__":

    def fx(x):
        return jnp.exp(jnp.tanh(x))

    def f(a, b):
        return jnp.matmul(a, b)

    import numpy as np
    from src.history.thesis_prototype_v2.intervals.main import RegisteredInterval

    ival = RegisteredInterval(1.0, 2.0)
    ivalA = np.asarray([[ival, ival],
                        [ival, ival]])
    ivalB = np.asarray([ival, ival])
    arguments = [ivalA, ivalB]

    # closed_jaxpr = jax.make_jaxpr(f)(*arguments)
    # FIXME: make the jax.make_jaxpr work with interval type input.
    closed_jaxpr = jax.make_jaxpr(f)(jnp.ones(1), jnp.ones(1))
    # otuput = IntervalJaxprEvaluator.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, jnp.ones(5))
    otuput = IntervalJaxprEvaluator.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *arguments)
    print(otuput)











