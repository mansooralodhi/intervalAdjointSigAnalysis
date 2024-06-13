
import jax.numpy as jnp
from jax import custom_jvp, grad, jvp

@custom_jvp
def sin(x):
    return jnp.sin(x)

@sin.defjvp
def f_jvp(primal, tangent):
    primal_out = sin(*primal)
    tangent_out = tangent[0] * jnp.cos(*primal)
    return primal_out, tangent_out

def f(x, y):
    return y * sin(x)


if __name__ == "__main__":
    primals, tangents = jvp(f, (2.0, 3.0), (1.0, 0.0))
    print(primals)
    print(tangents)
    print(f"{grad(f)(2.0, 3.0)}")

