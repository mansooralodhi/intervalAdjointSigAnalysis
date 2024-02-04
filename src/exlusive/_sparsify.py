import jax.numpy as jnp
from jax import random
from jax.experimental.sparse import BCOO, sparsify

mat = random.uniform(random.PRNGKey(1701), (5, 5))
mat = mat.at[mat < 0.5].set(0)
vec = random.uniform(random.PRNGKey(42), (5,))


mat_sparse = BCOO.fromdense(mat)

def f(mat, vec):
    return -(jnp.sin(mat) @ vec)

sparsify(f)(mat_sparse, vec)