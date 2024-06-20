

import jax
import jax.numpy as jnp
from jax import random


from src.interpreter.interpreter import Interpreter
from src.derivation_rules.vjp_rules.relu import relu

def parameters():
    key = random.key(0)
    keys = random.split(key, 3)
    x = random.normal(keys[0], shape=(10,))
    w1 = random.normal(keys[0], (5, 10))
    # print(w1)
    w2 = random.normal(keys[0], (5, 1))
    # print(w2)
    return x, w1, w2

def loss(params):
    x, w1, w2 = params
    return jnp.dot(relu(jnp.dot(w1, x)), w2)[0]

inputs = parameters()
output = loss(inputs)
grad = jax.grad(loss)(inputs)
print(grad)

interpret = Interpreter()
expr = jax.make_jaxpr(jax.grad(loss))(inputs)
out  = interpret.safe_interpret(expr.jaxpr, expr.literals, inputs)
# print(expr)
print(out)