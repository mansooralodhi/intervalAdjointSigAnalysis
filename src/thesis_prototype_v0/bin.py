from jax import Array

from mathematics.algo_differentiation.src_code_transformation.jax_implementation.grad import Grad
from mathematics.algo_differentiation.src_code_transformation.jax_implementation.utils import *

grad = Grad()

f1 = lambda x: x / 1
f2 = lambda x: x ** 4 - x ** 3 + x ** 2


def half(x):
    return x / 2


def polynomial(x):
    # first derivative = 4 x**3 - 3 x ** 2 + 2*x
    # second derivative = 12 x**2 - 6x + 2
    #   x = 1 : 3 :  8
    #   x = 2 :  32 - 12 + 4 = 24 : 48 - 12 +2 = 38
    return x ** 4 - x ** 3 + x ** 2


def dot_product(a: float | Array, b: float | Array) -> Array:
    return jax.numpy.vdot(a, b)


def custom_chain_rule(func, x: float | Array, v: float | Array):
    """
    important observation: we cannot compute derivatives of nested functions.
    if is impossible with simple functions to compute the derivative
    of nested functions w.r.t variable. Its is only possible with lexical
    closure or that is what algorithmic differentiation is used for that
    can do derivation using chain rule. it seems not an easy task to apply
    chain rule in python.
    question: if python run in sequential manner then why we cannot
    break the nested function into iterative steps ??????
    """
    inter1 = grad.func_gradient(func)
    y1 = grad.gradient_value(inter1, x)
    inter2 = grad.func_gradient(dot_product)
    y2 = grad.gradient_value(inter2, x, function_parameters=[v])
    return y2


if __name__ == "__main__":
    # ans = custom_chain_rule(x_cube, 2.0, 5.0)
    # print(ans)
    print(jax.grad(jax.grad(polynomial))(2.0))
    # print(jax.grad(f1, jax.grad(f2)(2.0))
