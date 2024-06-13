import jax
import jax.numpy as jnp
from jax import vjp, custom_vjp

@custom_vjp
def f(x, y):
    return y * jnp.sin(x)

def f_fwd(x, y):
    primal = f(x, y)
    residuals = (x, y, jnp.cos(x))
    return primal, residuals

def f_bwd(residuals, g):
    x, y, cos_x = residuals
    # Compute the gradients with respect to x and y
    # g is the cotangent passed from the chain rule (usually the gradient from subsequent layers)
    grad_x = g * cos_x * y
    grad_y = g * jnp.sin(x)
    return grad_x, grad_y

# register vjp rules
f.defvjp(f_fwd, f_bwd)


if __name__ == "__main__":
    primal, f_vjp = vjp(f, 1.0, 2.0)
    grad_x, grad_y = f_vjp(1.0)

    print(f"primal = {primal} ")
    print(f"adjoints  =  ({grad_x} , {grad_y} )")

    print(f"grad w.r.t x = {jax.grad(f,)(1., 2.0)}")