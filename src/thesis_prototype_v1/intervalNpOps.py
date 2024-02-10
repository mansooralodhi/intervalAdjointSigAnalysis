
import jax
import numpy as np
from intervalArithmetic import IntervalArithmetic

"""
Note: 
    -   Numpy primitive operation support Intervals (or pyTrees).
    -   Jax higher order transformation (i.e. jvp) do support (pyTrees) Intervals and
        model with overriden numpy operations..
"""

def create_array(x):
    print(np.asarray([x, x]))

def multiply_add(a, b, c):
    return np.add(np.dot(a, b), c)

def operator_multiply_add(a, b, c):
    return a * b  + c

def multiply_add_sin(a, b, c):
    return np.sin(np.add(np.dot(a, b), c))

def sample_jvp():
    # works
    func = lambda x: multiply_add(x,x,x)
    x = IntervalArithmetic(-1.0, 2.0)
    v = IntervalArithmetic(1.0, 0.0)
    print(jax.jvp(func, [x], [v]))

def sample_jacfwd():
    # works
    func = lambda x: multiply_add(x,x,x)
    x = IntervalArithmetic(-1.0, 2.0)
    v = IntervalArithmetic(1.0, 0.0)
    print(jax.jacfwd(func)(x))

if __name__ == "__main__":
    x = IntervalArithmetic(-1.0, 2.0)
    # create_array(x)
    # sample_jvp()
    sample_jacfwd()