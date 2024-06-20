import jax
import jax.numpy as jnp
from jax import custom_vjp

# Define the custom ReLU function
@custom_vjp
def custom_relu(x):
    return jnp.maximum(x, 0)

# Define the forward pass for the custom VJP
def custom_relu_fwd(x):
    return custom_relu(x), (x,)

# Define the backward pass for the custom VJP
def custom_relu_bwd(res, g):
    x, = res
    return (jnp.where(x > 0, g, 0),)

# Register the forward and backward passes with the custom ReLU function
custom_relu.defvjp(custom_relu_fwd, custom_relu_bwd)

# Test the custom ReLU function
x = jnp.array([-1.0, 0.0, 5.0, 15.0])
y = custom_relu(x)
print("Forward pass result:", y)

# Test the gradient of the custom ReLU function
grad_custom_relu = jax.grad(lambda x: jnp.sum(custom_relu(x)))
print("Gradient result:", grad_custom_relu(x))
