"""

Assumptions:
    -   intervals only interacts with intervals.
    -   lower bound of IntervalArithmetic is lower than greater bound

Operations:
    - addition
    - subtraction
    - multiplication

Source: https://jax.readthedocs.io/en/latest/pytrees.html#pytrees
"""

import numpy as np
from jax.tree_util import register_pytree_node_class
from jax.tree_util import tree_flatten, tree_unflatten


@register_pytree_node_class
class Interval(object):

    def __init__(self, lower_bound, upper_bound):
        # if not (type(lower_bound) is object or lower_bound is None or isinstance(lower_bound, Interval)):
        #     lower_bound = jnp.asarray(lower_bound)
        # if not (type(upper_bound) is object or upper_bound is None or isinstance(upper_bound, Interval)):
        #     upper_bound = jnp.asarray(upper_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @property
    def shape(self):
        return ()

    def __repr__(self):
        return "Interval(lower_bound={}, upper_bound={})".format(self.lower_bound, self.upper_bound)

    def __str__(self):
        return f"Interval({str(self.lower_bound)}, {str(self.upper_bound)})"

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

    def tree_flatten(self):
        children = (self.lower_bound, self.upper_bound)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

    @staticmethod
    def show_structure(yourTree):
        flat, tree = tree_flatten(yourTree)
        unflattened = tree_unflatten(tree, flat)
        print(f"{tree=}\n  {flat=}\n  {tree=}\n  {unflattened=}\n")


if __name__ == "__main__":
    # Interval.show_structure(Interval(3.0, 3.0))
    ivalA = Interval(5.0, 2.0)
    ivalB = Interval(-1.0, 8.0)
    print(np.add(ivalB, ivalA))
    print(ivalB + ivalA)

