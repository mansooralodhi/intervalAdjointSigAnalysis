import jax
from functools import wraps
from src.code.bin.makeJapr import make_jaxpr
from src.code.interpretJaxpr import interpret_jaxpr


def ivalTransformation(fun):
    # by Neil
    @wraps(fun)
    def wrapped(*args, intervals=None, **kwargs):
        closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
        print(closed_jaxpr.jaxpr)
        return interpret_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *intervals if intervals else args)[0]
    return wrapped


def _ivalTransformation(fun):
    # by Lodhi
    @wraps(fun)
    def wrapped(*args, **kwargs):
        expr = make_jaxpr(fun, *args, **kwargs)
        print(expr.jaxpr)
        return interpret_jaxpr(expr.jaxpr, expr.literals, *args)[0]
    return wrapped

