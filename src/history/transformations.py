import jax
from functools import wraps
from src.history.reinterpretJaxpr.interpretJaxpr import interpret_jaxpr


def ivalTransformation(fun):
    # by Neil
    @wraps(fun)
    def wrapped(*args, intervals=None, **kwargs):
        closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
        print(closed_jaxpr.jaxpr)
        return interpret_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *intervals if intervals else args)[0]
    return wrapped
