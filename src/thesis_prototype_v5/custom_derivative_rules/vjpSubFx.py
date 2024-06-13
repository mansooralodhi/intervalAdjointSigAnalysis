

import jax.numpy as jnp
from jax import custom_vjp, vjp, grad

"""
Note:   We cannot define custom_vjp rule for inner/nested function
        and leave the outer function uncustomized and use the 
        jax.vjp.
        Hence we customize the nested the function but use 
        the jax.grad on the outer function.
        
         
"""
@custom_vjp
def sin(x):
    return jnp.sin(x)

def sin_fwd(x):
    primal = sin(x)
    residual = (x, jnp.cos(x))
    return primal, residual

def sin_bwd(residuals, g):
    x, cos_x = residuals
    grad_x = g * cos_x
    return grad_x

sin.defvjp(sin_fwd, sin_bwd)

def f(x, y):
    return y * sin(x)


if __name__ == "__main__":
    # primals, f_vjp = vjp(f, 1.0, 2.0)
    # print(f"primals = {primals}")
    # print(f"adjoints = {f_vjp(1.0)}")
    print(grad(f, argnums=1)(1.0, 2.0))
