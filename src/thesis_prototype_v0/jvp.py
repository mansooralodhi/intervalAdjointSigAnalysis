import jax
from typing import Callable, Any

from jax import Array

from mathematics.algo_differentiation.src_code_transformation.jax_implementation.utils import vector_value_function


class JacobianVectorProduct(object):
    def __init__(self):
        pass

    @staticmethod
    def full_jacobian(f: Callable) -> Callable:
        return jax.jacfwd(f)

    @staticmethod
    def jacobian_vector_product(f: Callable, primals: jax.Array, tangents: jax.Array) -> tuple[Array, Array]:
        return jax.jacfwd(f)(primals), jax.jacfwd(f)(primals) @ tangents

    @staticmethod
    def jvp(f, primals: tuple, tangents: tuple) -> tuple[Any, ...]:
        return jax.jvp(f, primals,  tangents)


if __name__ == "__main__":
    eval_point = jax.numpy.asarray([1.0, 0.5, 1.5, 2.0])
    multi_point = jax.numpy.asarray([0.2, 0.3, 0.4, 0.8])

    jvp = JacobianVectorProduct()

    f_evaluated, jvp_evaluated = jvp.jvp(vector_value_function, (eval_point,), (multi_point,))
    print(f_evaluated)
    print(jvp_evaluated)

    jacobian, jvp_evaluated = jvp.jacobian_vector_product(vector_value_function, eval_point, multi_point)
    print(jacobian)
    print(jvp_evaluated)