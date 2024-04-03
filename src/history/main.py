import jax
import jax.numpy as np
from src.history.reinterpretJaxpr.utils import interval2scalarArgs
from src.code.transformations import ivalTransformation


def f(x1, x2, x3):
    """
    dfdx1 = lambda x1, x2, x3: 8 * x1 * x2 + 4 * x3 - x3 * x2 ** 2
    dfdx2 = lambda x1, x2, x3: 4 * x1 ** 2 - 2 * x1 * x2 * x3 - x3 ** 2
    dfdx3 = lambda x1, x2, x3: 4 * x1 - x1 * x2 ** 2 - x2 * x3 * 2
    """
    # return 4 * x2 * x1 ** 2 - x1 * x3 * x2 ** 2 + 4 * x1 * x3 - x2 * x3 ** 2
    # return ((4 * x1) - (x2 * x3))# * ((x1 * x2) + x3)
    return 4 * x1


ivalArgs = (np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([3.0, 4.0]))
scalarArgs = interval2scalarArgs(ivalArgs)

# print("f: ", ivalTransformation(f)(*scalarArgs, intervals=ivalArgs))


# print("jax grad w.r.t. x1: ", ivalTransformation(jax.grad(f))(*scalarArgs, intervals=ivalArgs))
print("jax grad w.r.t. x1: ", jax.make_jaxpr(ivalTransformation(jax.grad(f)))(*scalarArgs, intervals=ivalArgs))


# jit also works:
# jit_interval_f = jax.jit(ivalTransformation(jax.grad(f)))
# print("final jit grad w.r.t. x: ", jit_interval_f(*scalarArgs, intervals=ivalArgs))
# print("final jit reinterpretJaxpr: \n", jax.make_jaxpr(jit_interval_f)(x_orig, y_orig, intervals=(x_ival, y_ival)))
