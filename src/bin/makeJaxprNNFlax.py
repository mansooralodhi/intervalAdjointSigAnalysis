import jax
from flax import linen as nn
import flax.jax_utils as flax_utils

"""
Question: can we jax.make_jaxpr(NN) of neural network (NN) made using flax jax library ? 
Lets find out ... 

References:
    1. https://www.machinelearningnuggets.com/jax-cnn/#create-cnn-in-jax
"""

# Define your Flax neural network model
class MyModel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)
        x = nn.relu(x)
        return x

# Initialize model parameters
input_shape = (32, 64)  # Example input shape
rng_key = jax.random.PRNGKey(0)
params = MyModel(features=128).init(rng_key, jax.random.normal(rng_key, input_shape))

# Convert Flax model to a pure JAX function
model_fn = MyModel(features=128).apply
model_fn = jax.jit(model_fn)

# Generate JAXPR
input_data = jax.random.normal(rng_key, input_shape)
jaxpr = jax.make_jaxpr(model_fn)(params, input_data)
print(jaxpr)
