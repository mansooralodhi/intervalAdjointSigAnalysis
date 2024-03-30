import jax
import numpy as np
import jax.numpy as jnp
from jax.core import ShapedArray
from collections import namedtuple

Point = namedtuple(typename="Point", field_names=['x', 'y'])
point1 = Point(1.0, 2.0)

class Coordinate():
    def __init__(self, x, y):
        self.x, self.y = x, y

# point1 = Coordinate(1.0, 2.0)
# point1 = jnp.asarray(point1)


def func(a: Point, b: Point, c: Point):
    return jnp.multiply(jnp.add(a, b), c)

args = [[point1, point1, point1],
        [point1, point1, point1]]

args = np.asarray(args)
print(args)
print(jax.make_jaxpr(func)(args))