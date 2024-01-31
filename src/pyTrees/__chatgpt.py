import jax
import jax.numpy as np
from jax.tree_util import tree_flatten, tree_unflatten

class Special(object):
    def __init__(self, x, y):
        self.interval = [x, y]
        # self.y = y

    def __add__(self, other):
        return Special(self.interval[0] + other.interval[0],
                       self.interval[1] + other.interval[1])

    def __sub__(self, other):
        return Special(self.interval[0] - other.interval[0],
                       self.interval[1] - other.interval[1])

    # def __array__(self):
    #     return np.array([self.x, self.y], dtype=float)

# Define tree_flatten and tree_unflatten for Special
def tree_flatten_special(node):
    leaves = (node.interval[0], node.interval[1])
    aux = None  # Since Special has no additional auxiliary data
    return leaves, aux

def tree_unflatten_special(aux, leaves):
    return Special(*leaves)

# Register the PyTree structure for Special
jax.tree_util.register_pytree_node(
    Special, tree_flatten_special, tree_unflatten_special
)

# Now you can use Special with JAX numpy operations
special1 = Special(1, 2)
special2 = Special(3, 4)

# Example JAX operation using Special
result = jax.numpy.add(special1, special2)
print(result)  # Output: Special(x=4.0, y=6.0)
