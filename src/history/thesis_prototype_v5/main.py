"""
1. Define your function using jnp add, mul
2. Define interval range using jnp
3. Get jaxpr from make_jaxpr
4. Write interval jax registry
5. Use custom jaxpr interpreter
6. Find function output
7. Find derivative jax.grad, jax.jacfwd

Question:   what if jnp matrix has interval elements,
            what about Ax with intervals components?
            will tree_flatten be any helpful ?
Version 1:
Above code only accepts assumes interval inputs
Version 2:
Above code accepts both simple number and interval input
"""

import jax
import jax.numpy as jnp
from src.history.thesis_prototype_v5.ival import Interval
from src.history.thesis_prototype_v5.evalJaxpr import EvalJaxpr



# define function: x**2 + y**2
# assumes x and y are interval types
f = lambda x, y: jnp.add(jnp.multiply(x, x), jnp.multiply(y, y))

# define interval input
x = jnp.asarray([-4., -2.])
xIval = Interval(-4., -2.)
y = jnp.asarray([-1., 3.])
yIval = Interval(-1., 3.)

print("Basic Output: ", f(x, y))
output = jax.make_jaxpr(f)(x, y)
output = EvalJaxpr.eval_jaxpr(output.jaxpr, output.consts, x, y)
print("Interval Output: ", output[0])

import jax.numpy as jnp

