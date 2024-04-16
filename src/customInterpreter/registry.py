

import jax
from jax import lax
from typing import Dict
from jax._src.ad_util import add_jaxvals_p

# from src.customInterpreter.ops import *
from src.intervalArithmetic.ops import *

def ops_mapping() -> Dict:

    registry = dict()

    ################################# Recursive Operations ##############################
    # registry[custom_jvp_call_p] = jax.custom_jvp
    # registry[jax.experimental.pjit.pjit_p] = jax.experimental.pjit


    ################################### Static Operations ###############################
    registry[lax.device_put_p] = jax.device_put
    registry[add_jaxvals_p] = add_jaxvals_p.impl


    ################################### Custom Operations ###############################
    registry[lax.gt_p] = gt
    registry[lax.max_p] = max
    registry[lax.add_p] = add
    registry[lax.div_p] = divide
    registry[lax.select_n_p] = select_n
    registry[lax.reduce_sum_p] = reduce_sum
    registry[lax.dot_general_p] = dot_general
    registry[lax.broadcast_in_dim_p] = broadcast_in_dim

    ################################## Build-in Operations ##############################

    return registry

