
import jax
import jax.numpy as jnp
from jax import random, jit


from src.interpreter.interpreter import Interpreter
from src.derivation_rules.vjp_rules.relu import relu

interpret = Interpreter()

# def rostech(x1, x2, x3):
#     return (4.0 * x1 - x2 * x3) * (x1 * x2 + x3), (4.0 * x1 - x2 * x3) * (x1 * x2 + x3)

def rostech(x):
    return (4.0 * x[0] - x[1] * x[2]) * (x[0] * x[1] + x[2])

x = jnp.asarray([2.0, 4.0, 4.0])
print(rostech(x))
print('-' * 50)

expr = jax.make_jaxpr(rostech)(x)
print(interpret.safe_interpret(expr.jaxpr, expr.literals, [x]))
print('-' * 50)


y_grad = jax.grad(rostech)(x)
print(y_grad)
print('-' * 50)

expr = jax.make_jaxpr(jax.grad(rostech))(x)
print(expr)
t_grad = interpret.safe_interpret(expr.jaxpr, expr.literals, [x])
print(t_grad)
print('-' * 50)

