""""""

"""
dtype:  Interval
attributes: lower_bound
            upper_bound

NB: 
    1. jnp.ndarray can not posses interval dtypes.
    2. np.ndarray can posses interval dtype but are not traceable.
    3. overloaded np operations are not traceable for grad computation by jax.
    4. it is not possible to extend jnp.ndarray like np.ndarray.
    Hence,
        i. transform np.ndarray(dtype=interval) to jnp.ndarray(dtype=float)
        ii. re-define jax primitive ops (only) for derived jnp.ndarray 
        iii. transform f to make it executable with jax.grad
"""

import numpy as np

class Interval(np.ndarray):

    def __new__(cls, x, y):
        val = np.asarray([x, y], dtype=np.float32)
        obj = super().__new__(cls, shape=val.shape, buffer=val, dtype=np.float32)
        return obj


if __name__ == '__main__':
    ivals = Interval(2, 4)
    print(ivals)
    print(type(ivals))
    print(ivals.dtype)
    print(ivals.shape)
    print(ivals.size)
    print('*' * 35)

    ivals = np.asarray([ivals, ivals])#.view(Interval)
    print(ivals)
    print(type(ivals))
    print(ivals.dtype)
    print(ivals.shape)
    print(ivals.size)
    print('*' * 35)

    import jax.numpy as jnp
    ivals = Interval(2, 4)
    ivals = jnp.asarray([ivals, ivals])
    print(ivals)
    print(type(ivals))
    print(ivals.dtype)
    print(ivals.shape)
    print(ivals.size)
