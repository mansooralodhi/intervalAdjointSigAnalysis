
import jax
from jax import lax
from typing import Dict
from jax._src.ad_util import add_jaxvals_p


from src.interval_arithmetic.intervalArithmetic import IntervalArithmetic

ivalOps = IntervalArithmetic(jax.numpy)

def registry() -> Dict:

    registry = dict()

    ################################# Recursive Operations ##############################
    ###  Now handled by the custom interpreter !
    # registry[custom_jvp_call_p] = jax.custom_jvp
    # registry[jax.experimental.pjit.pjit_p] = jax.experimental.pjit

    ################################## Build-in Operations ##############################
    # registry[lax.ceil_p] = lax.ceil
    # registry[lax.expm1_p] = expm1

    ################################### Static Operations ###############################
    registry[lax.device_put_p] = jax.device_put
    registry[lax.convert_element_type_p] = ivalOps.convert_element_type

    ################################### Custom Operations ###############################
    registry[lax.neg_p] = ivalOps.negative
    registry[lax.add_p] = ivalOps.add
    registry[lax.sub_p] = ivalOps.subtract
    registry[lax.mul_p] = ivalOps.multiply
    registry[lax.div_p] = ivalOps.divide
    registry[lax.pad_p] = ivalOps.pad
    registry[lax.slice_p] = ivalOps.slice
    registry[lax.squeeze_p] = ivalOps.squeeze
    registry[lax.max_p] = ivalOps.maximum
    registry[lax.min_p] = ivalOps.minimum
    registry[lax.gt_p] = ivalOps.greater_than

    registry[add_jaxvals_p] = ivalOps.add
    registry[lax.sqrt_p] = ivalOps.sqrt
    registry[lax.transpose_p] = ivalOps.transpose
    registry[lax.reduce_sum_p] = ivalOps.sum
    registry[lax.dot_general_p] = ivalOps.dot_general
    registry[lax.broadcast_in_dim_p] = ivalOps.broadcast_in_dim


    return registry

