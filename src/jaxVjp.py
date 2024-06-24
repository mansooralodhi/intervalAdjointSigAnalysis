

import jax
import jax.numpy as jnp
from jax import random
from functools import partial, wraps

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
    w2 = random.normal(keys[1], (5, 2))
    return w1, w2

def loss(x, params):
    w1, w2 = params
    y = jnp.dot(relu(jnp.dot(w1, x)), w2)
    return y[0], y[1]


scalar_x, ival_x = inputs()
params = parameters()
y = loss(scalar_x, params)
print(y)
print("-"*50)
# prmals_out, vjp_func = jax.vjp(loss, scalar_x, params)
# t1 = vjp_func((1.0, 1.0))
# print("*" * 50)
# expr = jax.make_jaxpr(partial(jax.vjp, loss))(scalar_x, params)
# y = interpret.safe_interpret(expr.jaxpr, expr.literals, (scalar_x, *params))
# print(y)
# print()

expr = jax.make_jaxpr(jax.jacrev(loss, argnums=(0)))(scalar_x, params)
y2 = interpret.safe_interpret(expr.jaxpr, expr.literals, (ival_x, *params))
print(y2)
# print(all(y2[0] < y2[1]))
print(y2[0][0] < y2[0][1])
print(y2[1][0] < y2[1][1])
# expr = jax.make_jaxpr(jax.grad(loss2))(scalar_x, params)
# y = interpret.safe_interpret(expr.jaxpr, expr.literals, (scalar_x, *params))[0]
# y = interpret.safe_interpret(expr.jaxpr, expr.literals, (ival_x, *params))[0]
# print(all(y[0] < y[1]))
