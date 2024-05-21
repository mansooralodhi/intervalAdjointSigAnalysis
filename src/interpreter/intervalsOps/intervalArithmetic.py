from functools import partial
from typing import Union, Callable

import jax.lax

from src.interpreter.intervalsOps.bilinearfn import custom_bilinear
from src.interpreter.intervalsOps.numpyLike import (NDArray, NDArrayLike, Interval, IntervalLike, NumpyLike)

"""
Source: https://github.com/google/autobound/blob/3013a1030834b686f1bbb97ac9c2d825e51b0b7d/autobound/interval_arithmetic.py#L285
"""

import jax

class IntervalArithmetic():

    def __init__(self, np_like: NumpyLike):
        self.np_like = np_like

    ##################################### Transformation Operations ########################

    def as_interval(self, a: IntervalLike) -> Interval:
        return self.np_like.asarray(a[0]), self.np_like.asarray(a[1])

    def as_interval_or_ndarray(self, a: Union[NDArrayLike, IntervalLike]) -> Union[NDArray, Interval]:
        if isinstance(a, tuple):
            return self.as_interval(a)
        return self.np_like.asarray(a)

    def transpose(self, a: Union[NDArrayLike, IntervalLike], permutation=tuple) -> Union[NDArray, Interval]:
        if isinstance(a, tuple):
            return self.np_like.transpose(a[0], permutation), self.np_like.transpose(a[1], permutation)
        return self.np_like.transpose(a, permutation)

    ###################################### Attribute Operations ##############################

    def shape(self, a: Union[NDArrayLike, IntervalLike]) -> tuple:
        if isinstance(a, tuple):
            return self.np_like.shape(a[0])
        return self.np_like.shape(a)

    def ndim(self, a: Union[NDArrayLike, IntervalLike]) -> int:
        if isinstance(a, tuple):
            return self.np_like.ndim(a[0])
        return self.np_like.ndim(a)

    def convert_element_type(self, a: Union[NDArrayLike, IntervalLike], new_dtype, weak_type) -> int:
        if isinstance(a, tuple):
            return jax.lax.convert_element_type(a[0], new_dtype, weak_type), jax.lax.convert_element_type(a[1], new_dtype, weak_type)
        return jax.lax.convert_element_type(a, new_dtype)


    def expm1(self, a: Union[NDArrayLike, IntervalLike],) -> int:
        if isinstance(a, tuple):
            return self.np_like.expm1(a[0],), self.np_like.expm1(a[1], )
        return self.np_like.expm1(a,)
    ###################################### Condition Operations ##############################

    def maximum(self, a: Union[NDArrayLike, IntervalLike], b: NDArray) -> Union[NDArray, Interval]:
        if isinstance(a, tuple):
            if isinstance(b, NDArray) and b.shape == ():
                return self.np_like.maximum(a[0], b), self.np_like.maximum(a[1], b)
            else:
                raise NotImplementedError(b)
        return self.np_like.maximum(a, b)

    def greater_than(self, a: Union[NDArrayLike, IntervalLike], b: NDArray) -> Union[NDArray, Interval]:
        if isinstance(a, tuple):
            if isinstance(b, NDArray) and b.shape == ():
                return self.np_like.greater(a[0], b), self.np_like.greater(a[1], b)
            else:
                raise NotImplementedError(b)
        return self.np_like.greater(a, b)

    def choose(self, which: Union[NDArrayLike, IntervalLike], *cases) ->  Union[NDArrayLike, IntervalLike]:
        if isinstance(which, tuple):
            # fixme (fixed): this is to handle the special case of Relu activation function.
            #  Otherwise an improper interval is returned which don't follow the lb<up criteria.
            try:
                lb, ub = (self.np_like.choose(which[0].astype('int'), cases), self.np_like.choose(which[1].astype('int'), cases))
            except:
                print()
            # if all(lb <= ub):
            #     return lb, ub
            # if all(lb > ub):
            return ub, lb
            # raise Exception("Error: all lower bounds are not less than equal to upper bounds !!!!")
        return self.np_like.choose(which.astype('int'), cases)

    ####################################### Arithmetic Operations ############################

    def negative(self, a: Union[NDArrayLike, IntervalLike]) -> Union[NDArray, Interval]:
        if isinstance(a, tuple):
            return self.np_like.negative(a[1]), self.np_like.negative(a[0])
        return self.np_like.negative(a)

    def add(self, a: Union[NDArrayLike, IntervalLike], b: Union[NDArrayLike, IntervalLike]) -> Union[NDArray, Interval]:

        a_is_interval = isinstance(a, tuple)
        b_is_interval = isinstance(b, tuple)

        if a_is_interval and b_is_interval:
          return self.np_like.add(a[0], b[0]), self.np_like.add(a[1], b[1])

        elif a_is_interval:
          return self.np_like.add(a[0], b), self.np_like.add(a[1], b)

        elif b_is_interval:
          return self.np_like.add(a, b[0]), self.np_like.add(a, b[1])

        else:
          return self.np_like.add(a, b)

    def subtract(self, a: Union[NDArrayLike, IntervalLike], b: Union[NDArrayLike, IntervalLike]) -> Union[NDArray, Interval]:

        a_is_interval = isinstance(a, tuple)
        b_is_interval = isinstance(b, tuple)

        if a_is_interval and b_is_interval:
            return self.np_like.subtract(a[0], b[1]), self.np_like.subtract(a[1], b[0])

        elif a_is_interval:
            return self.np_like.subtract(a[0], b), self.np_like.subtract(a[1], b)

        elif b_is_interval:
            return self.np_like.subtract(a, b[1]), self.np_like.subtract(a, b[0])

        else:
            return self.np_like.subtract(a, b)

    def multiply(self, a: Union[NDArrayLike, IntervalLike], b: Union[NDArrayLike, IntervalLike]) -> Union[NDArray, Interval]:
        return self._arbitrary_bilinear(a, b, self.np_like.multiply, assume_product=True)

    def divide(self, a: Union[NDArrayLike, IntervalLike], b: NDArray) -> Union[NDArray, Interval]:
        b = self.np_like.divide(1, b)
        return self.multiply(a, b)

    def tensordot(self, a: Union[NDArrayLike, IntervalLike], b: Union[NDArrayLike, IntervalLike], axes) -> Union[NDArray, Interval]:
        bilinear = partial(self.np_like.tensordot, axes=axes)
        return self._arbitrary_bilinear(a, b, bilinear, assume_product = axes==0)

    def outer_product(self, a: Union[NDArrayLike, IntervalLike], b: Union[NDArrayLike, IntervalLike], batch_dims: int = 0) -> Union[NDArray, Interval]:
        """Interval variant of _ndarray_outer_product()."""
        # todo
        pass

    def _arbitrary_bilinear(self, a: Union[NDArrayLike, IntervalLike], b: Union[NDArrayLike, IntervalLike],
                            bilinear: Callable[[NDArrayLike, NDArrayLike], NDArrayLike], assume_product: bool=False) -> Union[NDArray, Interval]:

        a_is_interval = isinstance(a, tuple)
        b_is_interval = isinstance(b, tuple)

        if not a_is_interval and not b_is_interval:
            return bilinear(a, b)

        return custom_bilinear(a, b, bilinear, assume_product, self.np_like)

    ##################################### Log/Exp/Power Operations ############################

    def tanh(self, a: Union[NDArrayLike, IntervalLike]) -> Union[NDArray, Interval]:
        if isinstance(a, tuple):
            return self.np_like.tanh(a[0]), self.np_like.tanh(a[1])
        return self.np_like.tanh(a)

    def logistic(self, a: Union[NDArrayLike, IntervalLike]) -> Union[NDArray, Interval]:
        if isinstance(a, tuple):
            return jax.lax.logistic(a[0]), jax.lax.logistic(a[1])
        return jax.lax.logistic(a)

    def outer_power(self, a: Union[NDArrayLike, IntervalLike], exponent: int, batch_dims: int = 0) -> Union[NDArray, Interval]:
        """Returns a repeated outer product."""
        # todo
        pass

    def power(self, a: Union[NDArrayLike, IntervalLike], exponent: int, batch_dims: int = 0) -> Union[NDArray, Interval]:
        """Returns a**exponent (element-wise)."""
        # todo
        pass

    ##################################### Reduction/Expansion Operations #######################

    def sum(self, a: Union[NDArrayLike, IntervalLike], axis: int = 0):
        if isinstance(a, tuple):
            return self.np_like.sum(a[0], axis), self.np_like.sum(a[1], axis)
        return self.np_like.sum(a, axis)



if __name__ == "__main__":

    import numpy as np
    import jax.numpy as jnp

    a = (np.zeros((1, 3)), np.ones((1, 3)))
    b = (np.zeros((3,)), np.ones((3,)))
    ans =  (np.array([0.]), np.array([3.]))
    calculated = IntervalArithmetic(jnp).arbitrary_bilinear(a, b, partial(np.tensordot, axes=1), assume_product=False)
    print(calculated)
















