

import jax
import jax.numpy as jnp
from typing import Sequence, Optional, TypedDict, Dict

class Gradients(TypedDict):
    lowerBound: float
    upperBound: float

def jacobian2grad(jacobian: jnp.ndarray) -> Gradients:
    """
    jacobian -> (2, ...)
    """

    if not isinstance(jacobian, jnp.ndarray):
        raise Exception("Jacobian dtype not correct !")
    # todo:
    #  write checks for jacobian
    # lowerGrad = jacobian[0].sum().item()
    # upperGrad = jacobian[1].sum().item()
    lowerGrad = jacobian[0].ravel()[0].item()
    upperGrad = jacobian[1].ravel()[-1].item()
    return dict(lowerBound=lowerGrad, upperBound=upperGrad)


def generate_data(shape: Sequence[int], is_interval: bool = False) -> Optional[jnp.ndarray]:
    """
    if is_interval:
        return jnp.ndarray -> shape (..., 2)
    return jnp.ndarray --> shape (...)
    """
    if not shape:
        return
    if is_interval is False:
        key = jax.random.PRNGKey(0)
        val = jax.random.uniform(key, shape)
        return val
    shape = list(shape)
    shape.append(2)
    key = jax.random.PRNGKey(0)
    val = jax.random.uniform(key, shape)
    return val


if __name__ == '__main__':
    val = generate_data((2,2), True)
    print(val)
    print(val.dtype)
    print(val.shape)
    print(val.size)
