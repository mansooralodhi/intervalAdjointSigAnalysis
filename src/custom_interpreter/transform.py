


import jax
from typing import Callable
from functools import wraps
from src.custom_interpreter.interpreter import Interpreter

interpreter = Interpreter()

def transformation(fx: Callable):
    @wraps(fx)
    def wrapper(*scalarArgs, **kwargs):
        closedJaxpr = jax.make_jaxpr(fx)(*scalarArgs, **kwargs)
        return interpreter.safe_interpret(closedJaxpr.jaxpr, closedJaxpr.literals, *scalarArgs)[0]
    return wrapper

def interval_transformation(fx: Callable):
    # intervalize
    @wraps(fx)
    def wrapper(*scalarArgs, intervalArgs=None, **kwargs):
        closedJaxpr = jax.make_jaxpr(fx)(*scalarArgs, **kwargs)
        return interpreter.safe_interpret(closedJaxpr.jaxpr, closedJaxpr.literals, intervalArgs)
    return wrapper


def _vjp_interval_transformation(fx: Callable):
    @wraps(fx)
    def wrapper(*scalarArgs, intervalArgs=None, **kwargs):
        closedJaxpr = jax.make_jaxpr(fx)(*scalarArgs, **kwargs)
        return interpreter.safe_interpret(closedJaxpr.jaxpr, closedJaxpr.literals, *scalarArgs)
    return wrapper
