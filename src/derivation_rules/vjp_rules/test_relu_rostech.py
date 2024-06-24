import jax
import jax.numpy as jnp
from jax import random, jit

from src.interpreter.interpreter import Interpreter
from src.derivation_rules.vjp_rules.relu import relu
from src.interpreter.registry import registry
myRegistry = registry()

interpret = Interpreter()

import collections
from jax.core import  Literal


def custom_grad_interpreter(jaxpr, literals, inputs):
    env = collections.defaultdict(lambda: 0.0)

    def read(var):
        if isinstance(var, Literal):
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    # Initialize environment with inputs and literals
    for var, val in zip(jaxpr.invars, inputs + literals):
        write(var, val)

    from jax import lax
    # Process each equation
    for eqn in jaxpr.eqns:
        invals = [read(var) for var in eqn.invars]
        # if eqn.primitive == lax.reshape_p:
        #     print()
        # outvals = eqn.primitive.bind(*invals, **eqn.params)
        outvals = myRegistry[eqn.primitive](*invals, **eqn.params)
        if not isinstance(outvals, tuple):
            outvals = (outvals,)
        for var, val in zip(eqn.outvars, outvals):
            write(var, val)

    # Collect gradients
    grads = [env[var] for var in jaxpr.outvars]
    return grads

def rostech(x):
    x1, x2, x3 = x
    y1 = (4.0 * x1) - (x2 * x3)
    y2 = (x1 * x2) + x3
    y = y1 * y2
    return y, y


x = jnp.asarray([2.0, 4.0, 4.0])
expr = jax.make_jaxpr(jax.jacrev(rostech))(x)
print(expr)
t_grad = interpret.safe_interpret(expr.jaxpr, expr.literals, [(x, x+0.05)])
print("Custom Interpreter Gradient:", t_grad)