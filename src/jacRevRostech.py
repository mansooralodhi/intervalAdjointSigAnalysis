

import jax
import jax.numpy as jnp
from src.interpreter.interpreter import Interpreter
interpret = Interpreter()

def rostech(x1, x2, x3):
    return (4.0 * x1 - x2 * x3) * (x1 * x2 + x3)


x = jnp.asarray([2.0, 4.0, 4.0])
print(rostech(*x))


expr = jax.make_jaxpr(rostech)(*x)
print(interpret.safe_interpret(expr.jaxpr, expr.literals, [*x]))


y_grad = jax.jacrev(rostech)(*x)
print(y_grad)


expr = jax.make_jaxpr(jax.jacrev(rostech))(*x)
t_grad = interpret.safe_interpret(expr.jaxpr, expr.literals, [*x])
print(t_grad)


ival_x = [(1.0, 2.0), (3.0, 4.0), (3.0, 4.0)]
