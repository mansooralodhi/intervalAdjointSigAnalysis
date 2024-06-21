

import jax
import jax.numpy as jnp
from jax import random, jit


from src.interpreter.interpreter import Interpreter
from src.derivation_rules.vjp_rules.relu import relu

interpret = Interpreter()

def inputs():
    key = random.key(0)
    keys = random.split(key, 3)
    x = random.normal(keys[0], shape=(10,))
    x1 = random.normal(keys[1], shape=(10,))
    x2 = random.normal(keys[2], shape=(10,))
    print(all(x1<x2))
    return x, (x1, x2)

def parameters():
    key = random.key(1)
    keys = random.split(key, 2)
    w1 = random.normal(keys[0], (5, 10))
    w2 = random.normal(keys[1], (5, 1))
    return w1, w2

@jit
def loss(x, params):
    w1, w2 = params
    return jnp.dot(relu(jnp.dot(w1, x)), w2)[0]


x, x_ival = inputs()
params = parameters()

print(loss(x, params))

print(jax.grad(loss)(x, params))