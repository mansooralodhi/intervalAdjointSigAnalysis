
import jax.numpy as jnp


def my_reduce_sum(operand, axes):
    return jnp.sum(operand, axis=axes)
