

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
    print('-' * 50)

    return x, (x1, x2)

def parameters():
    key = random.key(1)
    keys = random.split(key, 2)
    w1 = random.normal(keys[0], (5, 10))
    w2 = random.normal(keys[1], (5, 2))
    return w1, w2

@jit
def loss(x, params):
    w1, w2 = params
    return jnp.dot(w2.T, relu(jnp.dot(w1, x)))


x, x_ival = inputs()
params = parameters()
print(loss(x, params))
print('-' * 50)
t_grad = jax.jacrev(loss)(x, params)

expr = jax.make_jaxpr(jax.jacrev(loss))(x, params)
y_grad = interpret.safe_interpret(expr.jaxpr, expr.literals, [x, *params])

print(t_grad)
print('-' * 50)

print(y_grad)
print('-' * 50)

y_grad = interpret.safe_interpret(expr.jaxpr, expr.literals, [x_ival, *params])
print(y_grad)
print('-' * 50)

y_grad = interpret.safe_interpret(expr.jaxpr, expr.literals, [x_ival[0], *params])

print(y_grad)
print('-' * 50)

y_grad = interpret.safe_interpret(expr.jaxpr, expr.literals, [x_ival[1], *params])

print(y_grad)
print('-' * 50)
