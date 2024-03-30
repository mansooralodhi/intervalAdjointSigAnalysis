from jax import lax
from jax._src.ad_util import add_jaxvals_p
from typing import Dict
from src.code.ivalOps import *


from jax._src.lax import lax

def get_registry() -> Dict:

    registry = dict()
    registry[lax.neg_p] = neg
    registry[lax.add_p] = add
    registry[lax.sub_p] = sub
    registry[lax.mul_p] = mul
    registry[lax.div_p] = div
    registry[lax.integer_pow_p] = integer_pow
    registry[add_jaxvals_p] = add_jaxvals_p.impl

    return registry
