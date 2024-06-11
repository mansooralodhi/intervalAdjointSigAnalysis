import jax
import jax.numpy as jnp

from functools import wraps

from jax import core
from jax import lax
from jax._src.util import safe_map
from src.thesis_prototype_v4.buildJaxpr import BuildJaxpr

interval_registry = {}

# etc...

class Interval:
    def __init__(self, x, y):
        self.range = jnp.asarray([x, y])

def interval_exp(x):
    if isinstance(x, Interval):
        pass
    elif isinstance(x, jnp.ndarray):
        if x.shape == (2,):
            return jnp.array([jnp.exp(x[0]), jnp.exp(x[1])])
        elif x.shape == ():
            return jnp.array(jnp.exp(x))
    else:
        raise NotImplementedError("interval mul not correctly implemented")

def interval_add(x, y):
    # TODO: put the real mul in here
    if isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):
        return x + y
    elif isinstance(x, float) and isinstance(y, Interval):
        # return Interval(x+y.range[0], y.range[1])
        return jnp.asarray([x + y.range[0], y.range[1]])
    elif isinstance(x, jnp.ndarray) and isinstance(y, Interval):
        # return Interval(x+y.range[0], y.range[1])
        return jnp.asarray([x[0] + y.range[0], x[1] + y.range[1]])
    else:
        raise NotImplementedError("interval mul not correctly implemented")


def interval_mul(x, y):
    # TODO: put the real mul in here
    if isinstance(x, float) and isinstance(y, jnp.ndarray):
        return x * y
    elif isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):
        return jnp.asarray([x[0] * y[0] , x[1] * y[1]])
    elif isinstance(x, float) and isinstance(y, Interval):
        # return Interval(x+y.range[0], y.range[1])
        return jnp.asarray([x * y.range[0], y.range[1]])
    else:
        raise NotImplementedError("interval mul not correctly implemented")


interval_registry[lax.add_p] = interval_add
interval_registry[lax.mul_p] = interval_mul
interval_registry[lax.exp_p] = interval_exp


def myinterval(fun):
    @wraps(fun)
    def wrapped(*args, intervals=None, **kwargs):
        # Since we assume unary functions, we won't worry about flattening and
        # unflattening arguments.
        # closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
        # out = interval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, intervals if intervals else args)
        jaxpr, _ = BuildJaxpr.build(f, *args, **kwargs)
        print(jaxpr)
        print("*" * 50)
        out = interval_jaxpr(jaxpr, [], intervals if intervals else args)
        return out

    return wrapped

def interval(fun):
    @wraps(fun)
    def wrapped(*args, intervals=None, **kwargs):
        # Since we assume unary functions, we won't worry about flattening and
        # unflattening arguments.
        closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
        print(closed_jaxpr.jaxpr)
        out = interval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, intervals if intervals else args)
        return out

    return wrapped


def interval_jaxpr(jaxpr, consts, *args):
    env = {}

    def read(var):
        if type(var) is core.Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    safe_map(write, jaxpr.invars, *args)
    safe_map(write, jaxpr.constvars, consts)

    for eqn in jaxpr.eqns:
        invals = safe_map(read, eqn.invars)

        if eqn.primitive not in interval_registry:
            raise NotImplementedError(
                f"{eqn.primitive} does not have registered interval equivalent.")

        outvals = interval_registry[eqn.primitive](*invals)
        safe_map(write, eqn.outvars, [outvals])

    return safe_map(read, jaxpr.outvars)


def f(x, y):
    # return jnp.exp(2 * x + y)
    return jnp.exp(x + y)


f_int = interval(f)

# we run on scalar input to produce scaler output
# because if we have to calculate grad we need a scalar output.
x_orig = jnp.array(0.0)
y_orig = jnp.array(1.0)

x_int = jnp.array([0.0, 1.0])
y_int = jnp.array([1.0, 2.0])
# x_int = Interval(0.0, 1.0)
# y_int = Interval(1.0, 2.0)

print(myinterval(jax.grad(f, [0, 1]))(x_int, y_int))
print(interval(jax.grad(f, 0))(x_orig, y_orig))
# print(jax.grad(f, 0)(x_int, y_int))
# print("final result: ", interval(f)(x_orig, y_orig, intervals=(x_int, y_int)))
print("final grad w.r.t. x: ", interval(jax.grad(f, 0))(x_orig, y_orig, intervals=(x_int, y_int)))
print("final grad w.r.t. y: ", interval(jax.grad(f, 1))(x_orig, y_orig, intervals=(x_int, y_int)))
# print("final jaxpr (unoptimized): \n", jax.make_jaxpr(interval(jax.grad(f)))(x_orig, y_orig, intervals=(x_int, y_int)))

# jit also works:

# jit_interval_f = jax.jit(interval(jax.grad(f)))
# print("final jit grad w.r.t. x: ", jit_interval_f(x_orig, y_orig, intervals=(x_int, y_int)))
# print("final jit jaxpr: \n", jax.make_jaxpr(jit_interval_f)(x_orig, y_orig, intervals=(x_int, y_int)))
