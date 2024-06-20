
import jax
from jax import nn
import jax.numpy as jnp
from jax import custom_vjp

# todo: modify it to work for intervals as well as scalar inputs


@custom_vjp
def relu(x):
    return jnp.maximum(x, 0)

def relu_fwd(x):
    primal = relu(x)
    residual = jnp.asarray(jnp.greater(x, 0.0), dtype=jnp.float32)
    return primal, residual

def relu_bwd(residual, g):
    return g * residual,

relu.defvjp(relu_fwd, relu_bwd)


if __name__ == "__main__":

    def fnn(x):
        return jnp.sum(nn.relu(x))

    def f(x):
        return jnp.sum(relu(x))

    x = jnp.array([-1.0, 0.0, 5.0, 15.0])
    print(f"x  =  {x}")
    print(f"f(x) nn.relu = {fnn(x)}")
    print(f"f(x)  = {f(x)}")
    print(f"grad(f) (nn.relu) = {jax.grad(fnn)(x)}")
    print(f"grad(f) (relu) = {jax.grad(f)(x)}")