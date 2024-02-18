import jax
from jax import core
import jax.numpy as jnp
from jax._src.util import safe_map
from registry import inverse_registry

class InverseJaxprEvaluator(object):

    env = {}

    @classmethod
    def read(cls, var):
        if type(var) is core.Literal:
            return var.val
        print(f"Reading from env: var= {var}, val= {cls.env[var]}")
        return cls.env[var]

    @classmethod
    def write(cls, var, val):
        print(f"Writing in env: var= {var}, val= {val}")
        cls.env[var] = val

    @classmethod
    def eval_jaxpr(cls, jaxpr, consts, *args):
        # Args now correspond to Jaxpr outvars

        safe_map(cls.write, jaxpr.outvars, args)
        safe_map(cls.write, jaxpr.constvars, consts)

        # Looping backward
        for eqn in jaxpr.eqns[::-1]:
            #  outvars are now invars
            invals = safe_map(cls.read, eqn.outvars)
            if eqn.primitive not in inverse_registry:
                raise NotImplementedError(
                    f"{eqn.primitive} does not have registered inverse.")
            """ Assuming a unary function """
            outval = inverse_registry[eqn.primitive](*invals)
            safe_map(cls.write, eqn.invars, [outval])
        output = safe_map(cls.read, jaxpr.invars)
        cls.env = {}
        return output


if __name__ == "__main__":
    def f(x):
      return jnp.exp(jnp.tanh(x))


    closed_jaxpr = jax.make_jaxpr(f)(jnp.ones(5))
    otuput = InverseJaxprEvaluator.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, jnp.ones(5))
    print(otuput)
    # assert jnp.allclose(f_inv(f(1.0)), 1.0)