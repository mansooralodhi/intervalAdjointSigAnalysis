import jax
import jax.numpy as jnp


# Define your neural network model
def neural_network(params, x):
    # Example neural network computation
    hidden = jnp.dot(x, params['W1']) + params['b1']
    output = jnp.dot(hidden, params['W2']) + params['b2']
    return output


# Define your loss function
def loss_fn(params, x, y):
    predictions = neural_network(params, x)
    return jnp.mean((predictions - y) ** 2)


# Compute gradients of the loss function with respect to intermediate values
@jax.jit
def intermediate_gradients(params, x, y):
    # Forward pass to compute intermediate values
    hidden = jnp.dot(x, params['W1']) + params['b1']
    output = jnp.dot(hidden, params['W2']) + params['b2']

    # Compute gradients of the loss function with respect to intermediate values
    grads_hidden = jax.grad(loss_fn, argnums=1)(params, x, y)
    grads_output = jax.grad(loss_fn, argnums=2)(params, x, y)

    return grads_hidden, grads_output


# Example usage
params = {'W1': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
          'b1': jnp.array([1.0, 2.0]),
          'W2': jnp.array([[5.0], [6.0]]),
          'b2': jnp.array([3.0])}

x = jnp.array([[1.0, 2.0]])
y = jnp.array([[10.0]])

grads_hidden, grads_output = intermediate_gradients(params, x, y)
print("Gradients w.r.t. hidden layer:", grads_hidden)
print("Gradients w.r.t. output layer:", grads_output)
