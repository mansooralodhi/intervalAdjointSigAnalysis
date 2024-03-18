

from jax import lax
from typing import Dict
from src.code.ops import *

def get_registry() -> Dict:

    registry = dict()


    registry[lax.add_p] = add
    registry[lax.sub_p] = sub
    registry[lax.exp_p] = exp
    registry[lax.mul_p] = mul
    registry[lax.div_p] = div
    registry[lax.reduce_sum_p] = reduce_sum
    registry[lax.reduce_max_p] = reduce_max
    return registry