import jax
import jax.numpy as jnp
from jax import custom_jvp

@custom_jvp
def relu(x):
    return jnp.maximum(x, 0)


@relu.defjvp
def relu_tangent(primals, tangents):
    (x,) = primals
    (t,) = tangents
    y = relu(x)
    y_dot = t * jnp.asarray(jnp.greater(x, 0.0), dtype=jnp.float32)  # ReLU derivative is 1 if x > 0, else 0
    return y, y_dot



if __name__ == "__main__":
    # Test the custom JVP implementation
    x = jnp.array([-1.0, 0.0, 1.0])
    t = jnp.array([1.0, 1.0, 1.0])

    # Compute the JVP directly
    primal, tangent = jax.jvp(relu, (x,), (t,))
    print("Primal output:", primal)
    print("Tangent output:", tangent)

    # Compute the gradient to verify it works
    grad_custom_relu = jax.grad(lambda x: jnp.sum(relu(x)))
    print("Gradient output:", grad_custom_relu(x))
