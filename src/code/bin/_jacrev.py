


import jax.numpy as jnp
from jax import jacrev, grad, jacfwd

def jacobian2grad(jacob):
    # lowerGrad = jacob[0].ravel()[0]
    # upperGrad = jacob[1].ravel()[-1]
    lowerGrad = jacob[0].sum().item()
    upperGrad = jacob[1].sum().item()
    return jnp.asarray([lowerGrad, upperGrad])

def f(a, b):
    y = a*b
    return jnp.asarray([jnp.sum(y[..., 0]), jnp.sum(y[..., 1])])
    # y2 = a * b
    # return jnp.asarray([y1, y2])


ival = jnp.asarray([5.0, 3.0])
x = jnp.asarray([[ival, ival]])
y = jnp.asarray([[ival, ival]])
print(x.shape)
print(f(x, y))
print(f(x, y).shape)
J = jacrev(f)(x,  y)
print(J.shape)
print(jacobian2grad(J))
print(J)