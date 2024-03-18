import jax
import jax.numpy as jnp
from src.code.utils import jacobian2grad
from src.code.ivalArithmetic import ivalArtihmeticByLodhi


def f(x1, x2, x3):
    return (4 * x1 - x2 * x3) * (x1 * x2 + x3)


x1 = jnp.array([1.0, 2.0])
x2 = jnp.array([3.0, 4.0])
x3 = jnp.array([3.0, 4.0])

output = ivalArtihmeticByLodhi(f)(x1, x2, x3)
print("f: ", output)


# jacobian_wrt_x = jax.jacrev(ivalArtihmeticByLodhi(f), 0)(x1, x2, x3)
# intervalGrads_wrt_x = jacobian2grad(jacobian_wrt_x)
# jacobian_wrt_y = jax.jacrev(ivalArtihmeticByLodhi(f), 1)(x1, x2, x3)
# intervalGrads_wrt_y = jacobian2grad(jacobian_wrt_y)
#
# print("final grad w.r.t. x: ", intervalGrads_wrt_x)
# print("final grad w.r.t. y: ", intervalGrads_wrt_y)

############################################### END #######################################################

# fixme:
#       this transformation doesn't work because
#       the output is a vector (interval) and grad
#       doesn't work with vector output.
# jax.grad(interval_transform(f), 0)(x, y)

# fixme:
#       the below transformation produce same result as
#       original function output, it doesn't help compute
#       gradient w.r.t any variable.
# (interval_arithmetic(jax.grad(f, 0))(A, B)
