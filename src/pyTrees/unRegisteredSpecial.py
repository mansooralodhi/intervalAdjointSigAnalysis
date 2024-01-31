

import jax
from special import Special

jax.tree_util.tree_leaves([
    Special(0, 1),
    Special(2, 4),
])

try:
    jax.tree_map(lambda x: x + 1,
    [
      Special(0, 1),
      Special(2, 4),
    ])
except TypeError as e:
    print(f'TypeError: {e}')

"""
    This is because if you define your own container class, it will be 
considered to be a pytree leaf unless you register it with JAX.
    Accordingly, if you try to use a jax.tree_map() expecting the leaves 
to be elements inside the container, you will get an error: 
    TypeError: unsupported operand type(s) for +: 'Special' and 'int'

"""