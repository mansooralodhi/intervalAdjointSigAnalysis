import jax.numpy
from jax import random
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import List


class Model(object):

    def __init__(self, num_input_variables: int):
        self.inputVariables = num_input_variables
        self.model_parameters = dict(W=ArrayLike, b=ArrayLike)
        self.initialize_model_params()
        self.static_x = None

    def predict(self, x: ArrayLike = None, W: ArrayLike = None, b: ArrayLike = None) -> jnp.float32 | List[jnp.float32]:
        """
        scalar-value function
        Args:
            x: inputs to model with size (no_of_samples, no_of_variables)
            W: weight matrix of linear layers with size (1, no_of_variables)
            b: float type
        Returns:
            A value or vector (depends upon x if it's a 'vector' with single instance
            or 'matrix' with batch)
        """
        # todo: set the below condition for wrapper function above predict()
        if x is None:
            x = self.static_x
        if W is None:
            W = self.model_parameters.get("W")
        if b is None:
            b = self.model_parameters.get("b")
        return self.sigmoid(self.primary_function(x, W, b))

    def loss(self, x: ArrayLike, y: ArrayLike, W: ArrayLike, b: ArrayLike) -> jax.numpy.float32:
        """
        scalar-value function
        Args:
            x: inputs to model with size (no_of_samples, no_of_variables)
            y: boolean labels to input with size (no_of_samples,)
            W: weight matrix (1, no_of_variables)
            b: float type
        Returns:
            A value or vector (depends upon x if it's a 'vector' with single instance
            or 'matrix' with batch)
        """
        return self.negative_log_likelihood(self.sigmoid(self.primary_function(x, W, b)), y)

    def initialize_model_params(self):
        key = random.PRNGKey(0)
        W_key, b_key, key = random.split(key, 3)
        self.model_parameters["W"] = random.normal(W_key, (self.inputVariables,))
        self.model_parameters["b"] = random.normal(b_key, ())

    @staticmethod
    def primary_function(x: ArrayLike, W: ArrayLike, b: ArrayLike) -> jnp.float32 | List[jnp.float32]:
        """
        scalar-value function
        Args:
            x: inputs to model with size (no_of_samples, no_of_variables)
            W: weight matrix (1, no_of_variables)
            b: float type
        Returns:
            A value or vector (depends upon x if it's a 'vector' with single instance
            or 'matrix' with batch)
        """
        return jnp.dot(x, W) + b

    @staticmethod
    def sigmoid(x: ArrayLike) -> ArrayLike:
        """
        vector-value function
        Args:
            x: Array [float32]
        Returns:
            y: Array [float32] -> x.shape

        however, since sigmoid is being used after primary_function therefore it acts as
        scalar-value function
        """
        return 0.5 * (jnp.tanh(x / 2) + 1)

    @staticmethod
    def negative_log_likelihood(y_label: ArrayLike, y_model: ArrayLike) -> jnp.float32:
        """
        scalar-value function
        Args:
            y_label: Array [float32]
            y_model: Array [float32]
        Returns:
            scalar value type float32
        """
        label_probs = y_model * y_label + (1 - y_model) * (1 - y_label)
        return -jnp.sum(jnp.log(label_probs))


if __name__ == "__main__":
    model = Model(4)
    model.initialize_model_params()

    from data import Data

    data = Data(4)
    x = data.generate_X(2)
    y = data.generate_Y(2)
    ans = model.loss(x, y, *model.model_parameters.values())
    print(ans)
