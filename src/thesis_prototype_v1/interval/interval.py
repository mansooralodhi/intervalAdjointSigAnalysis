import numpy as np
import jax.numpy as jnp

class Interval(object):

    def __init__(self, lower_bound, upper_bound):
        if not (type(lower_bound) is object or lower_bound is None or isinstance(lower_bound, Interval)):
            lower_bound = jnp.asarray(lower_bound)
        if not (type(upper_bound) is object or upper_bound is None or isinstance(upper_bound, Interval)):
            upper_bound = jnp.asarray(upper_bound)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __repr__(self):
        return "IntervalArithmetic(lower_bound={}, upper_bound={})".format(self.lower_bound, self.upper_bound)

    def __str__(self):
        return f"[{str(self.lower_bound)}, {str(self.upper_bound)}]"

    def __add__(self, other):
        return Interval(self.lower_bound + other.lower_bound, self.upper_bound + other.upper_bound)

    def __sub__(self, other):
        return Interval(self.lower_bound - other.upper_bound, self.upper_bound - other.lower_bound)

    def __mul__(self, other):
        lower_bound = min(self.lower_bound * other.lower_bound, self.lower_bound * other.upper_bound,
                          self.upper_bound * other.lower_bound, self.upper_bound * other.upper_bound)
        upper_bond = max(self.lower_bound * other.lower_bound, self.lower_bound * other.upper_bound,
                         self.upper_bound * other.lower_bound, self.upper_bound * other.upper_bound)
        return Interval(lower_bound, upper_bond)

    def tanh(self):
        # can be used with numpy
        return Interval(np.tanh(self.lower_bound), np.tanh(self.upper_bound))

    def exp(self):
        # can be used with numpy
        return Interval(np.exp(self.lower_bound), np.exp(self.upper_bound))
