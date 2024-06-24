
import jax.numpy as jnp
from jax import random

# from src.site_packages.derivation_rules.vjp_rules.relu import relu
from src.site_packages.derivation_rules.jvp_rules.relu import relu


def inputs():
    key = random.key(0)
    keys = random.split(key, 3)
    x = random.normal(keys[0], shape=(10,))
    x1 = random.normal(keys[1], shape=(10,))
    x2 = random.normal(keys[2], shape=(10,))
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
    return y
