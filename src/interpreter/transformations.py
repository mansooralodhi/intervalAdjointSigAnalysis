


import jax
from typing import Callable
from functools import wraps
from src.interpreter.interpreter import Interpreter

"""
The below transformations act as an interface to use the custom interpreter (K) for computing the 
scalar and interval datatypes primals and adjoints  !
"""
interpreter = Interpreter()

def scalar_primal_transformation(fx: Callable):
    @wraps(fx)
    def wrapper(*scalarArgs, **kwargs):
        closedJaxpr = jax.make_jaxpr(fx)(*scalarArgs, **kwargs)
        return interpreter.safe_interpret(closedJaxpr.jaxpr, closedJaxpr.literals, *scalarArgs)[0]
    return wrapper

def scalar_adjoint_transformation(fx: Callable):
    @wraps(fx)
    def wrapper(*scalarArgs, **kwargs):

        closedJaxpr = jax.make_jaxpr(fx)(*scalarArgs, **kwargs)
        return interpreter.safe_interpret(closedJaxpr.jaxpr, closedJaxpr.literals, *scalarArgs)[0]
    return wrapper

def interval_primal_transformation(fx: Callable):
    @wraps(fx)
    def wrapper(*scalarArgs, intervalArgs=None, **kwargs):
        closedJaxpr = jax.make_jaxpr(fx)(*scalarArgs, **kwargs)
        return interpreter.safe_interpret(closedJaxpr.jaxpr, closedJaxpr.literals, intervalArgs)
    return wrapper

def interval_adjoint_transformation(fx: Callable):
    @wraps(fx)
    def wrapper(*scalarArgs, intervalArgs=None, **kwargs):
        closedJaxpr = jax.make_jaxpr(fx)(*scalarArgs, **kwargs)
        return interpreter.safe_interpret(closedJaxpr.jaxpr, closedJaxpr.literals, intervalArgs)
    return wrapper


def interval_vjp_transformation(fx: Callable):
    # fixme: unable to use intervals as primals.
    @wraps(fx)
    def wrapper(*scalarArgs, intervalArgs=None, **kwargs):
        closedJaxpr = jax.make_jaxpr(fx)(*scalarArgs, **kwargs)
        return interpreter.safe_interpret(closedJaxpr.jaxpr, closedJaxpr.literals, *scalarArgs)
    return wrapper
