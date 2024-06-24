

import jax
from functools import wraps
from typing import Callable

from src.site_packages.custom_interpreter.interpreter import safe_interpret

from typing import Sequence
from src.site_packages.interval_arithmetic.numpyLike import Interval, NDArray

def scalar_interpret(f: Callable):
    @wraps(f)
    def wrapper(scalar_input: NDArray, model_parameters: Sequence[NDArray]):
        jaxExpr = jax.make_jaxpr(f)(scalar_input, model_parameters)
        output = safe_interpret(jaxExpr.jaxpr, jaxExpr.literals, scalar_input, *model_parameters)
        return output
    return wrapper


def interval_interpret(f: Callable):
    @wraps(f)
    def wrapper(interval_input: Interval, scalar_input: NDArray, model_parameters: Sequence[NDArray]):
        jaxExpr = jax.make_jaxpr(f)(scalar_input, model_parameters)
        output = safe_interpret(jaxExpr.jaxpr, jaxExpr.literals, interval_input, *model_parameters)
        return output
    return wrapper


