import jax
import jax.numpy as np
from intervalArithmetic import IntervalArithmetic

"""
Note: 
    -   Jax primitive operation (using jax.numpy) don't support Intervals (or pyTrees).
    -   Jax higher order transformation (i.e. jvp) do support (pyTrees) Intervals .
    -   By default, intervals are not valid jax types, they are not traceable. 
    
Question:   
        How do we implement the model if jax primitive operations don't support 
        intervals ? We cannot only rely on arithmetic operators only.

Possible Solution:
    -   Replace jax primitive operations with numpy operations.
    -   Write new jax primitives in numpy and use them in jax.
    -   Rewrite jax interpreters. 
"""


def create_array(x):
    # not works
    print(np.asarray([x, x]))


def multiply_add(a, b, c):
    # not works
    return np.add(np.dot(a, b), c)


def operator_multiply_add(a, b, c):
    # works
    return a * b + c


def multiply_add_sin(a, b, c):
    # not works
    return np.sin(np.add(np.dot(a, b), c))


def sample_jvp():
    # works
    # doesn't work with func = lambda x: multiply_add(x,x,x)
    # func = lambda x: multiply_add(x, x, x)
    func = lambda x: operator_multiply_add(x, x, x)
    x = IntervalArithmetic(-1.0, 2.0)
    v = IntervalArithmetic(1.0, 0.0)
    print(jax.jvp(func, [x], [v]))


def sample_jacfwd():
    # works
    # doesn't work with func = lambda x: multiply_add(x,x,x)
    # func = lambda x: multiply_add(x, x, x)
    func = lambda x: operator_multiply_add(x, x, x)
    x = IntervalArithmetic(-1.0, 2.0)
    # v = IntervalArithmetic(1.0, 0.0)
    # print(jax.jacfwd(func)(x))
    print(jax.grad(func)(x))


if __name__ == "__main__":
    # x = IntervalArithmetic(-1.0, 2.0)
    # y = operator_multiply_add(x, x, x)
    sample_jacfwd()
