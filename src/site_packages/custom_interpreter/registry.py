
import jax
from jax import lax
from typing import Dict
from jax._src.ad_util import add_jaxvals_p


from src.site_packages.interval_arithmetic.intervalArithmetic import IntervalArithmetic

ivalOps = IntervalArithmetic(jax.numpy)

def registry() -> Dict:

    registry = dict()

    ################################# Recursive Operations ##############################
    ###  Now handled by the custom custom_interpreter !
    # registry[custom_jvp_call_p] = jax.custom_jvp
    # registry[jax.experimental.pjit.pjit_p] = jax.experimental.pjit

    ################################## Build-in Operations ##############################
    # registry[lax.ceil_p] = lax.ceil
    # registry[lax.expm1_p] = expm1

    ################################### Static Operations ###############################
    registry[lax.device_put_p] = jax.device_put

    ################################### Custom Operations ###############################
    registry[lax.add_p] = ivalOps.add
    registry[add_jaxvals_p] = ivalOps.add
    registry[lax.broadcast_in_dim_p] = ivalOps.broadcast_in_dim
    registry[lax.div_p] = ivalOps.divide
    registry[lax.convert_element_type_p] = ivalOps.convert_element_type
    registry[lax.dot_general_p] = ivalOps.dot_general
    registry[lax.eq_p] = ivalOps.equal
    registry[lax.gt_p] = ivalOps.greater_than
    registry[lax.iota_p] = ivalOps.iota
    registry[lax.max_p] = ivalOps.maximum
    registry[lax.min_p] = ivalOps.minimum
    registry[lax.mul_p] = ivalOps.multiply
    registry[lax.neg_p] = ivalOps.negative
    registry[lax.pad_p] = ivalOps.pad
    registry[lax.reduce_sum_p] = ivalOps.sum
    registry[lax.reshape_p] = ivalOps.reshape
    registry[lax.slice_p] = ivalOps.slice
    registry[lax.select_n_p] = ivalOps.choose
    registry[lax.squeeze_p] = ivalOps.squeeze
    registry[lax.sub_p] = ivalOps.subtract
    registry[lax.sqrt_p] = ivalOps.sqrt
    registry[lax.transpose_p] = ivalOps.transpose


    return registry

