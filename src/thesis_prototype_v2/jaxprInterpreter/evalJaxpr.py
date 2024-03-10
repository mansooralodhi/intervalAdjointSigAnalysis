import jax
import jax.numpy as jnp
from functools import wraps
from src.thesis_prototype_v2.jaxprInterpreter.inverse import InverseJaxprEvaluator
from src.thesis_prototype_v2.jaxprInterpreter.default import DefaultJaxprEvaluator


# from src.jaxprInterpreter.intervalJaxprEvaluator import Inter

class EvaluateJaxpr(object):

    @staticmethod
    def inverse_jaxpr(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            closed_jaxpr = jax.make_jaxpr(function)(*args, **kwargs)
            out = InverseJaxprEvaluator().eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
            return out[0]

        return wrapper

    @staticmethod
    def default_jaxpr(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            closed_jaxpr = jax.make_jaxpr(function)(*args, **kwargs)
            out = DefaultJaxprEvaluator.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
            return out[0]

        return wrapper

    @staticmethod
    def interval_jaxpr(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            closed_jaxpr = jax.make_jaxpr(function)(*args, **kwargs)
            out = DefaultJaxprEvaluator.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
            return out[0]

        return wrapper


if __name__ == "__main__":
    def f(x):
        return jnp.exp(jnp.tanh(x))


    evaljaxpr = EvaluateJaxpr()
    f_inv = evaljaxpr.inverse_jaxpr(f)
    print(f_inv.__name__)
    assert jnp.allclose(f_inv(f(1.0)), 1.0)
