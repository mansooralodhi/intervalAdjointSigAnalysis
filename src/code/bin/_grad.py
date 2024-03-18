import jax
from jax import lax
from jax import core
import jax.numpy as jnp
from functools import wraps
from jax._src.util import safe_map
from src.code.bin._buildjaxpr import BuildJaxpr


interval_registry = {}

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

# etc...

def interval_exp(x):
    # TODO: put the real mul in here
    if x.shape == ():
        return jnp.array(jnp.exp(x))
    elif x.shape == (2,):
        return jnp.array([jnp.exp(x[0]), jnp.exp(x[1])])
    raise NotImplementedError("this exp case not implemented !")

def interval_add(x, y):
    # TODO: put the real mul in here
    return x + y

def interval_mul(x, y):
    # TODO: put the real mul in here
    return x[0] * y[0]

interval_registry[lax.add_p] = interval_add
interval_registry[lax.mul_p] = interval_mul
interval_registry[lax.exp_p] = interval_exp


def intervalByLodhi(fun):
    @wraps(fun)
    def wrapped(*args, intervals=None, **kwargs):
        jaxpr, _ = BuildJaxpr.build(f, *args, **kwargs)
        out = interval_jaxpr(jaxpr, [], intervals if intervals else args)
        return out
    return wrapped

def intervalByNeil(fun):
    @wraps(fun)
    def wrapped(*args, intervals=None, **kwargs):
        closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
        print(closed_jaxpr.jaxpr)
        out = interval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, intervals if intervals else args)
        return out
    return wrapped

####################################### Objective Function ###################################

def f(x, y):
    # return jnp.exp(2 * x + y)
    # return jnp.exp(x + y)
    return x * y

####################################### Input Arguments ###################################

x_ival = jnp.array([[5.0, 10.0], [5.0, 10.0]])
y_ival = jnp.array([[3.0, 7.0], [3.0, 7.0]])

print("*" * 30, "  Neil Implementation ", " *" * 30)
# we run on scalar input to produce scaler output
# because if we have to calculate grad we need a scalar output.
x_orig = jnp.array(0.0)
y_orig = jnp.array(1.0)

# the below doesn't work bcz of vector output (2,) )
# print(jax.grad(f, 0)(x_ival, x_ival))

print("final result: ", intervalByNeil(f)(x_orig, y_orig, intervals=(x_ival, y_ival)))
print("final grad w.r.t. x: ", intervalByNeil(jax.grad(f, 0))(x_orig, y_orig, intervals=(x_ival, y_ival)))
print("final grad w.r.t. y: ", intervalByNeil(jax.grad(f, 1))(x_orig, y_orig, intervals=(x_ival, y_ival)))
# print("final jaxpr (unoptimized): \n", jax.make_jaxpr(intervalByNeil(jax.grad(f)))(x_orig, y_orig, intervals=(x_ival, x_ival)))

# jit also works:
jit_interval_f = jax.jit(intervalByNeil(jax.grad(f)))
print("final jit grad w.r.t. x: ", jit_interval_f(x_orig, y_orig, intervals=(x_ival, y_ival)))
# print("final jit jaxpr: \n", jax.make_jaxpr(jit_interval_f)(x_orig, y_orig, intervals=(x_ival, y_ival)))

# print("*"*30, "  Mansoor Implementation ", " *"*30)
# print(intervalByLodhi(jax.grad(f, 0))(x_ival, y_ival))
