import jax
import jax.numpy as jnp
from src.code.ivalArithmetic import ivalArithmeticByNeil


def f(x1, x2, x3):
    return (4 * x1 - x2 * x3) * (x1 * x2 + x3)


x1 = jnp.array([1.0, 2.0])
x2 = jnp.array([3.0, 4.0])
x3 = jnp.array([3.0, 4.0])

#######################################   Neils Implementation ########################################

print("*" * 30, "  Neil Implementation ", " *" * 30)
x_orig = jnp.asarray(11.0)
y_orig = jnp.asarray(12.0)
z_orig = jnp.asarray(12.0)

print("final result: ", ivalArithmeticByNeil(f)(x_orig, y_orig, z_orig, intervals=(x1, x2, x3)))
print("final grad w.r.t. x: ", ivalArithmeticByNeil(jax.grad(f, 0))(x_orig, y_orig, z_orig, intervals=(x1, x2, x3)))
print("final grad w.r.t. y: ", ivalArithmeticByNeil(jax.grad(f, 1))(x_orig, y_orig, z_orig, intervals=(x1, x2, x3)))
# print("final jaxpr (unoptimized): \n", jax.make_jaxpr(intervalByNeil(jax.grad(f)))(x_orig, y_orig, intervals=(
# x_ival, x_ival)))

# jit also works:
jit_interval_f = jax.jit(ivalArithmeticByNeil(jax.grad(f)))
print("final jit grad w.r.t. x: ", jit_interval_f(x_orig, y_orig, z_orig, intervals=(x1, x2, x3)))
# print("final jit jaxpr: \n", jax.make_jaxpr(jit_interval_f)(x_orig, y_orig, intervals=(x_ival, y_ival)))
