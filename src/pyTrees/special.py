
import jax

class Special(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Special(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Special(self.x - other.y, self.y - other.x)

    def __mul__(self, other):
        x = min(self.x * other.x, self.x * other.y,
                          self.y * other.x, self.y * other.y)
        y = max(self.x * other.x, self.x * other.y,
                         self.y * other.x, self.y * other.y)
        return Special(x, y)

    def __array__(self):
        return jax.numpy.array([self.x, self.y])