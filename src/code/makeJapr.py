"""
"""

"""
Input: 
    float | jnp.ndarray

NB: jnp.ndarray could have been derived from interval or np.ndarray(dtype=interval)
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Dict


def make_jaxpr(func: Callable, *args, **kwargs):
    args, kwargs = _drop_last_axis(args, kwargs)
    return jax.make_jaxpr(func)(*args, **kwargs)


def _drop_last_axis(args: Tuple, kwargs: Dict) -> Tuple[list, dict]:
    finalArgs = list()
    finalKwargs = dict()
    for arg in args:
        if isinstance(arg, jnp.ndarray):
            finalArgs.append(arg[..., 0])
        else:
            finalArgs.append(arg)
    for key, val in kwargs.items():
        if isinstance(val, jnp.ndarray):
            finalKwargs[key] = val[..., 0]
        else:
            finalKwargs[key] = val
    return finalArgs, finalKwargs


if __name__ == "__main__":
    f = lambda x: jnp.exp(x)
    x = jnp.asarray([[2, 3]])
    expr = make_jaxpr(f, x)
    print(expr)
