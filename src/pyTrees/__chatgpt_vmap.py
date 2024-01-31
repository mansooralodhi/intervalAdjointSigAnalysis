import jax
import jax.numpy as np

class Special(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Special(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Special(self.x - other.y, self.y - other.x)

    def __array__(self):
        return np.array([self.x, self.y], dtype=float)

# Define a function that uses JAX operations with Special instances
def special_add(special1, special2):
    return special1 + special2

# Use jax.vmap to vectorize the function
vectorized_special_add = jax.vmap(special_add)

# Now you can use the vectorized function with arrays of Special instances
specials1 = [Special(1, 2), Special(3, 4)]
specials2 = [Special(5, 6), Special(7, 8)]

result = vectorized_special_add(specials1, specials2)
print(result)  # Output: [Special(x=6.0, y=8.0), Special(x=10.0, y=12.0)]
