
import jax

"""
NB:
    1.  Scalar-valued function
    2.  Minimum two-layer neural network (including input-output layer)
    3.  Each layer includes bias
"""

class DenseNN(object):
    def __init__(self):
        self.layers = [264, 128, 1]
        self.weights = list()
        self.init_weights()

    @staticmethod
    def relu(x):
        return jax.numpy.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1.0/(1.0+jax.numpy.exp(-x))

    def init_weights(self):
        keys = jax.random.split(jax.random.key(0), len(self.layers)-1)
        for i in range(1, len(self.layers)):
            b = jax.random.normal(key=keys[i-1], shape=(self.layers[i],))
            w = jax.random.normal(key=keys[i-1], shape=(self.layers[i], self.layers[i-1]))
            self.weights.append([w, b])

    def calculate_y(self, x):
        y = jax.numpy.dot(x, self.weights[0][0].T) + self.weights[0][1]
        for i in range(1, len(self.weights)):
            y = self.sigmoid(y)
            y = jax.numpy.dot(y, self.weights[i][0].T) + self.weights[i][1]
        return y[0]

    @staticmethod
    def calculate_y_active(x, weights):
        y = jax.numpy.dot(x, weights[0].T) + weights[1]
        y = jax.numpy.maximum(0, y)
        y = jax.numpy.dot(y, weights[2].T) + weights[3]
        y = jax.numpy.maximum(0, y)
        y = jax.numpy.dot(y, weights[4].T) + weights[5]
        return y[0]


if __name__ == "__main__":
    network = DenseNN()
    network.init_weights()
    x = jax.random.normal(jax.random.key(0), shape=(264,))
    y = network.calculate_y(x)
    print(y)




