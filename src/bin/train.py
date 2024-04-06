import jax
import jax.numpy as jnp
from flax import linen
from jax import random, grad
from collections import defaultdict
from tensorflow.keras.datasets import mnist

# Source: https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html

####################################### Data ####################################
# x_train:  (60000, 28, 28) | y_train:  (60000,)
# x_test:   (10000, 28, 28) | y_train:  (10000,)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = x_train.reshape(-1, 28 * 28), x_test.reshape(-1, 28 * 28)

################################### Hyperparameters ############################
p: int | tuple = 784    # features size
n: int = 10             # number of labels
lr: float = 0.02        # learning rate
b: int = 32             # batch size
epochs: int = 20        # epochs

####################################### NN #####################################
class ANN(linen.Module):
    @linen.compact
    def __call__(self, x):       # Dense: x @ w
        x = linen.Dense(128)(x)  # (..., p) * (p, 128,) = (..., 128)
        x = linen.relu(x)        # (128)
        x = linen.Dense(n)(x)    # (..., 128) * (128 * n) * (128) = (n)
        return x

##################################### Initialize NN ###########################
model = ANN()
key = random.PRNGKey(0)
inputSample = jnp.ones(p)
model_params_dict = model.init(key, inputSample)

##################################### Loss Function ###########################
def loss(params_dict, x, y):
    """
    Args:
        params_dict: {'params': ...}
        x: (b, n) | (b, ...)
        y: (b,)
    Return:
        scalar
    """
    scores = model.apply(params_dict, x)    # (b, n)
    y = jax.nn.one_hot(y, n)                # (b, n)
    y_score = scores * y                    # (b, n)
    y_score = linen.log_softmax(y_score)    # (b, n)
    y_loss = jnp.sum(y_score, axis=-1)      # (b,)
    tot_loss = -jnp.mean(y_loss)            # ()
    return tot_loss

#################################### Score #############################
def predict(params_dict, x,):
    """
    Args:
        params_dict: {'params': ...}
        x: (b, n) | (b, ...)
        y: (b,)
    Return:
        predicted_label: (b,)
    """
    scores = model.apply(params_dict, x)         # (b, n)
    max_score_lbl = jnp.argmax(scores, axis=-1)  # (b,)
    return max_score_lbl

############################### Update Model Parameters #################
def update_params(params_dict, gradient):
    params = params_dict.get('params')
    gradient = gradient.get('params')
    updated_params = defaultdict(lambda: defaultdict(dict))
    for layer in params:
        for para in params[layer]:
            updated_params['params'][layer][para] = params[layer][para] - lr * gradient[layer][para]
    return updated_params

##################################### Accuracy ###########################
def accuracy(params_dict, x, y):
    """
    Args:
        params_dict: {'params': ...}
        x: (b, n) | (b, ...)
        y: (b,)
    Return:
        scalar
    """
    estimate = predict(params_dict, x)
    return jnp.mean(estimate == y) * 100

#################################### Train Step ###########################
def train_step(params_dict, x, y):
    gradient = grad(loss)(params_dict, x, y)
    updated_param = update_params(params_dict, gradient)
    return updated_param

#################################### Train Step ###########################
def train():
    for epoch in range(epochs):
        for i in range(0, len(x_train), 32):
            x = jnp.asarray(x_train[i:i+32])
            y = jnp.asarray(y_train[i:i+32])
            model_params_dict = train_step(model_params_dict, x, y)

################################### Final Test ###########################
# test_accu = accuracy(model_params_dict, jnp.asarray(x_test), jnp.asarray(y_test))
# print(test_accu)
expr = jax.make_jaxpr(loss)(model_params_dict, x_train[0], y_train[0])
print(expr)
