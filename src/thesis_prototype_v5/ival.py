import jax.numpy as jnp


class Interval(jnp.asarray):
    range: jnp.ndarray

    def __init__(self, x: float, y: float):
        super().__init__(shape=(2,))
        self.range = jnp.asarray([x, y])


val = Interval(1, 2)
print(val.range)
val.range = jnp.asarray([1., 2, 3])
print(val.range)
print(isinstance(val, Interval))
