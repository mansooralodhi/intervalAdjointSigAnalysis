
import jax
from jax import lax
from typing import Dict
from jax._src.ad_util import add_jaxvals_p

from src.interpreter.ops import *



def ops_mapping() -> Dict:

    registry = dict()

    ################################# Recursive Operations ##############################
    # registry[custom_jvp_call_p] = jax.custom_jvp
    # registry[jax.experimental.pjit.pjit_p] = jax.experimental.pjit


    ################################### Static Operations ###############################
    registry[lax.device_put_p] = jax.device_put
    registry[add_jaxvals_p] =  add # problem with: add_jaxvals_p.impl
    # registry[lax.ceil_p] = lax.ceil



    ################################### Custom Operations ###############################
    registry[lax.gt_p] = gt
    registry[lax.max_p] = max
    registry[lax.add_p] = add
    registry[lax.sub_p] = sub
    registry[lax.mul_p] = mul
    registry[lax.div_p] = divide
    registry[lax.tanh_p] = tanh
    registry[lax.select_n_p] = select_n
    registry[lax.transpose_p] = transpose
    registry[lax.reduce_sum_p] = reduce_sum
    registry[lax.dot_general_p] = dot_general
    registry[lax.broadcast_in_dim_p] = broadcast_in_dim

    ################################## Build-in Operations ##############################

    return registry

