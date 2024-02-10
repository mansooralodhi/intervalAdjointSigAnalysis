from mathematics.algo_differentiation.src_code_transformation.jax_implementation.model import Model
from mathematics.algo_differentiation.src_code_transformation.jax_implementation.data import Data

import jax
from typing import Callable, Tuple, List
from jax.test_util import check_grads


class Grad(object):
    def __init__(self):
        pass

    @staticmethod
    def func_gradient(func: Callable, partial_variables: int | Tuple[int] = 0) -> Callable:
        """
        Args:
            func: scalar value functions
            partial_variables: index of partial variable to compute derivative
        Returns:
            func: derivative of func w.r.t variables partial_variables
        """
        return jax.grad(func, argnums=partial_variables)

    @staticmethod
    def gradient_value(gradient_func: Callable, x: float | jax.typing.ArrayLike,
                       function_parameters: dict | List = None) \
            -> jax.typing.ArrayLike:
        """
        Args:
            gradient_func: gradient of scalar value function
            x: sample / instance / point -> (num_weights, )
            function_parameters: keys and values of variables to function
        Return:
            value of partial derivative of function at point x
        """
        if not isinstance(x, float):
            assert len(x.shape) == 1, "Error: gradient only possible for scalar output !"
        if function_parameters is None:
            return gradient_func(x)
        if type(function_parameters) == list:
            return gradient_func(x, *function_parameters)
        return gradient_func(x, *function_parameters.values())

    def fun_gradient_value(self, func, x: float | jax.typing.ArrayLike, partial_variables: int | Tuple[int] = 0,
                           function_parameters: dict | List = None) -> jax.typing.ArrayLike:
        """
        Args:
            func: scalar value functions
            x: sample / instance / point -> (num_weights, )
            partial_variables: index of partial variable to compute derivative
            function_parameters: keys and values of variables to function
        Return:
            value of partial derivative of function at point x
        """
        return self.gradient_value(self.func_gradient(func, partial_variables), x, function_parameters)

    @staticmethod
    def value_grad(func: Callable, partial_variables: int | Tuple[int], function_parameters: dict | List = None) \
            -> Tuple[jax.numpy.float32, Callable]:
        """
        Args:
            func: scalar value function
            partial_variables: index of variables w.r.t compute derivative
            function_parameters: keys and values of variables to function
        Return:
            (value of func, derivative of func)
        """
        if function_parameters is None:
            return jax.value_and_grad(func, partial_variables)()
        if type(function_parameters) == list:
            return jax.value_and_grad(func, partial_variables)(*function_parameters)
        return jax.value_and_grad(func, partial_variables)(*function_parameters.values())

    @staticmethod
    def verify_grad(func, x: jax.typing.ArrayLike, function_parameters: dict | List = None) -> None:
        """
        Args:
            func: gradient of a function
            x: sample / instance / point
            function_parameters: keys and values of variables to function
        Return:
            error if the grad obtained at x doesn't match with numerical (finite) difference at x
        """
        if function_parameters is None:
            print(check_grads(func, x, order=2))
        if type(function_parameters) == list:
            print(check_grads(func, (x, *function_parameters), order=2))
        if type(function_parameters) == dict:
            print(check_grads(func, (x, *function_parameters.values()), order=2))


if __name__ == "__main__":
    num_input_variables = 2
    batch_size = 2
    model = Model(num_input_variables)
    dataset = Data(num_input_variables)
    diff = Grad()
    print(diff.fun_gradient_value(model.primary_function, dataset.generate_X(), 0, model.model_parameters))
