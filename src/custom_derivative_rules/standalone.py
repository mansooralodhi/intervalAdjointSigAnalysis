
import jax.numpy as jnp
from jax import random as rand


def x_active():
    k = rand.key(0)
    keys = rand.split(k, 5)
    w1 = rand.normal(keys[0], shape=(10, 15))
    w2 = rand.normal(keys[1], shape=(10, 5))
    b1 = rand.normal(keys[2], shape=(10,))
    b2 = rand.normal(keys[3], shape=(5,))
    x = rand.normal(keys[4], shape=(15,))
    return x, w1, b1, w2, b2

def loss(x):
    x = x[0]
    x_in, w1, b1, w2, b2 = x
    v0 = jnp.dot(w1, x_in) + b1
    v1 = jnp.dot(v0, w2) + b2
    v3 = jnp.linalg.norm(v1)
    return v3


if __name__ == "__main__":
    w = x_active()
    y = loss(w)
    print(y)
