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
from jax.tree_util import tree_flatten, tree_unflatten
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class IntervalArithmetic(object):

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __repr__(self):
        return "IntervalArithmetic(lower_bound={}, upper_bound={})".format(self.lower_bound, self.upper_bound)

    def __str__(self):
        return f"[{str(self.lower_bound)}, {str(self.upper_bound)}]"

    def __add__(self, other):
        return IntervalArithmetic(self.lower_bound + other.lower_bound, self.upper_bound + other.upper_bound)

    def __sub__(self, other):
        return IntervalArithmetic(self.lower_bound - other.upper_bound, self.upper_bound - other.lower_bound)

    def __mul__(self, other):
        lower_bound = min(self.lower_bound * other.lower_bound, self.lower_bound * other.upper_bound,
                          self.upper_bound * other.lower_bound, self.upper_bound * other.upper_bound)
        upper_bond = max(self.lower_bound * other.lower_bound, self.lower_bound * other.upper_bound,
                         self.upper_bound * other.lower_bound, self.upper_bound * other.upper_bound)
        return IntervalArithmetic(lower_bound, upper_bond)

    def sin(self):
        # can be used with numpy
        return self

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
    IntervalArithmetic.show_structure(IntervalArithmetic(3.0, 3.0))
