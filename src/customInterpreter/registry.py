



import jax
from jax import lax
from typing import Dict
# from jax._src.lax import lax

from src.customInterpreter.ops import *


# source: https://github.com/google/jax/blob/9a931af6664e04ef97bb243ad2a4f7aa7290290b/jax/custom_derivatives.py#L28
# source: https://github.com/google/jax/blob/9a931af6664e04ef97bb243ad2a4f7aa7290290b/jax/_src/ad_util.py#L37
from jax._src.ad_util import add_jaxvals_p

def ops_mapping() -> Dict:

    registry = dict()
    registry[lax.gt_p] = lax.gt
    registry[lax.max_p] = lax.max
    registry[lax.add_p] = jax.numpy.add
    registry[lax.div_p] = jax.numpy.divide
    registry[lax.select_n_p] = lax.select_n
    registry[lax.reduce_sum_p] = my_reduce_sum
    registry[lax.device_put_p] = jax.device_put
    registry[add_jaxvals_p] = add_jaxvals_p.impl
    registry[lax.dot_general_p] = lax.dot_general
    registry[lax.broadcast_in_dim_p] = lax.broadcast_in_dim

    # todo: write a recursive interpret to interpret these below operations.
    # registry[custom_jvp_call_p] = jax.custom_jvp
    # registry[jax.experimental.pjit.pjit_p] = jax.experimental.pjit

    # registry[lax.neg_p] = neg
    # registry[lax.sub_p] = sub
    # registry[lax.mul_p] = mul
    # registry[lax.div_p] = div
    # registry[lax.integer_pow_p] = integer_pow

    return registry

