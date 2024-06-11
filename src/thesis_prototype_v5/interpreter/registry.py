from jax import lax
from typing import Dict

from src.thesis_prototype_v5.interpreter.opsInterface import *

def ops_mapping() -> Dict:

    registry = dict()

    ################################# Recursive Operations ##############################
    # registry[custom_jvp_call_p] = jax.custom_jvp
    # registry[jax.experimental.pjit.pjit_p] = jax.experimental.pjit

    ################################# Static/Built-in Operations ###############################
    # registry[add_jaxvals_p] = add  # problem with: add_jaxvals_p.impl
    # registry[lax.expm1_p] = expm1
    # registry[lax.ceil_p] = lax.ceil
    # registry[lax.device_put_p] = jax.device_put
    # registry[lax.convert_element_type_p] = convert_element_type

    ################################### Custom Operations ###############################
    registry[lax.gt_p] = gt
    registry[lax.eq_p] = eq
    registry[lax.max_p] = max
    registry[lax.min_p] = min
    registry[lax.select_n_p] = select_n

    registry[lax.add_p] = add
    registry[lax.sub_p] = sub
    registry[lax.mul_p] = mul
    registry[lax.div_p] = divide
    registry[lax.neg_p] = neg
    registry[lax.dot_general_p] = dot_general
    registry[lax.integer_pow_p] = integer_pow

    registry[lax.tanh_p] = tanh
    registry[lax.logistic_p] = logistic
    registry[lax.exp_p] = exp

    registry[lax.transpose_p] = transpose
    registry[lax.broadcast_in_dim_p] = broadcast_in_dim
    registry[lax.reduce_sum_p] = reduce_sum
    registry[lax.slice_p] = slice
    registry[lax.squeeze_p] = squeeze
    registry[lax.pad_p] = pad


    return registry

