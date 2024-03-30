from jax import lax
from intervalArithmetic import IntervalArithmetic

"""
Note: 
    -   lax primitive operations doesn't intervals or pytree-intervals because it is not traceable.
"""
def multiply_add_lax(x, y, z):

  return lax.add(lax.mul(x, y), z)


if __name__ == "__main__":
    x = IntervalArithmetic(1.0, 2.0)
    y = multiply_add_lax(x,x,x)
    print(y)