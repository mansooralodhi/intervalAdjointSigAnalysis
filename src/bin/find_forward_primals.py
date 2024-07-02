
import jax
from jax import random
import jax.numpy as jnp
from src.site_packages.custom_interpreter.interpreter import safe_interpreter

key = random.key(0)
keys = random.split(key, 3)
x = random.normal(keys[0], (15,))
w1 = random.normal(keys[1], (15, 7))
w2 = random.normal(keys[2], (7, 2))

f = lambda x, params: jnp.dot(jnp.maximum(jnp.dot(x, params[0]), 0), params[1])


jaxExpr = jax.make_jaxpr(f)(x, (w1, w2))
print(jaxExpr.jaxpr)
output, intermediates = safe_interpreter(jaxExpr.jaxpr, jaxExpr.literals, x, *(w1, w2))
print(output)