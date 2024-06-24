

import jax
from functools import wraps
from typing import Callable, Sequence
from src.site_packages.custom_interpreter.interpreter import safe_interpreter
from src.site_packages.interval_arithmetic.numpyLike import Interval, NDArray


# fixme:    lets stick to this functional api to access the custom interpreter
#           and not modify the safe_interpreter arguments, so for this purpose
#           lets say we have a model with inputs and some parameters.
#           later we may modify the below functions s.t they work even if model
#           does not have parameters.

# todo, Note: unlike make_jaxexpr the safe_interpreter needs flatten arguments.


def scalar_interpret(f: Callable):
    @wraps(f)
    def wrapper(scalar_input: NDArray, model_parameters: Sequence[NDArray]):
        jaxExpr = jax.make_jaxpr(f)(scalar_input, model_parameters)
        output = safe_interpreter(jaxExpr.jaxpr, jaxExpr.literals, scalar_input, *model_parameters)
        return output
    return wrapper


def interval_interpret(f: Callable):
    @wraps(f)
    def wrapper(interval_input: Interval, scalar_input: NDArray, model_parameters: Sequence[NDArray]):
        jaxExpr = jax.make_jaxpr(f)(scalar_input, model_parameters)
        output = safe_interpreter(jaxExpr.jaxpr, jaxExpr.literals, interval_input, *model_parameters)
        return output
    return wrapper


