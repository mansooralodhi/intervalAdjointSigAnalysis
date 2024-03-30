import jax
import numpy as np
from functools import wraps
from src.history.thesis_prototype_v4.interval import Interval
from src.history.thesis_prototype_v4.buildJaxpr import BuildJaxpr
from src.history.thesis_prototype_v4.evalJaxpr import EvalJaxpr

BuildJaxpr.customDtype.append(Interval)

def transform_function(fun):
    @wraps(fun)
    def wrapped(*args):
        jaxpr, consts = BuildJaxpr.build(fun, *args)
        output = EvalJaxpr.eval_jaxpr(jaxpr, consts, *args)
        return output[0]
    return wrapped

f1 = lambda x, y: x * y
f2 = lambda x, y: np.multiply(x, y)
ival = Interval(1., 2.)
# args = [ival, np.asarray([[ival, ival], [ival, ival]])]
args = [ival, ival]
# print(transform_function(f1)(*args))

print(jax.jacfwd(transform_function(f2), [0, 1])(*args))
# jax.grad(transform_function(f1))(*args)
# jax.grad(f)(*args)

