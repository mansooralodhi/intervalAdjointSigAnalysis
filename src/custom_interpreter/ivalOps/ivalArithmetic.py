from functools import partial
from typing import Union, Callable

from src.custom_interpreter.ivalOps import utils
from src.custom_interpreter.ivalOps.bilinearfn import custom_bilinear
from src.custom_interpreter.ivalOps.numpyLike import (NDArray, NDArrayLike, Interval, IntervalLike, NumpyLike)

"""
Source: https://github.com/google/autobound/blob/3013a1030834b686f1bbb97ac9c2d825e51b0b7d/autobound/interval_arithmetic.py#L285
"""


class IntervalArithmetic():

    def __init__(self, np_like: NumpyLike):
        self.np_like = np_like

    @staticmethod
    def _is_interval(a) -> bool:
        return isinstance(a, tuple)

    def _as_interval(self, a: IntervalLike) -> Interval:
        return self.np_like.asarray(a[0]), self.np_like.asarray(a[1])

    def _as_interval_or_ndarray(self, a: Union[NDArrayLike, IntervalLike]) -> Union[NDArray, Interval]:
        if isinstance(a, tuple):
            return self._as_interval(a)
        return self.np_like.asarray(a)

    ###################################### Attribute Operations ##############################

    def shape(self, a: Union[NDArrayLike, IntervalLike]) -> tuple:
        if self._is_interval(a):
            return self.np_like.shape(a[0])
        return self.np_like.shape(a)

    def ndim(self, a: Union[NDArrayLike, IntervalLike]) -> int:
        if self._is_interval(a):
            return self.np_like.ndim(a[0])
        return self.np_like.ndim(a)

    ###################################### Conditional Operations #############################

    def maximum(self, a: Union[NDArrayLike, IntervalLike], b: NDArrayLike) -> Union[NDArray, Interval]:
        if self._is_interval(a) and isinstance(b, NDArrayLike):
            return self.np_like.maximum(a[0], b), self.np_like.maximum(a[1], b)
        return self.np_like.maximum(a, b)

    def minimum(self, a: Union[NDArrayLike, IntervalLike], b: NDArrayLike) -> Union[NDArray, Interval]:
        if self._is_interval(a) and isinstance(b, NDArrayLike):
            return self.np_like.minimum(a[0], b), self.np_like.minimum(a[1], b)
        return self.np_like.minimum(a, b)

    def greater_than(self, a: Union[NDArrayLike, IntervalLike], b: NDArrayLike) -> Union[NDArray, Interval]:
        if self._is_interval(a) and isinstance(b, NDArrayLike):
            return self.np_like.greater(a[0], b), self.np_like.greater(a[1], b)
        return self.np_like.greater(a, b)

    def choose(self, which: Union[NDArray, Interval], *cases: tuple) -> Union[NDArrayLike, IntervalLike]:
        if self._is_interval(which):
            # fixme (fixed): this is to handle the special case of Relu activation function.
            #  Otherwise an improper interval is returned which don't follow the lb<up criteria.
            lb, ub = (self.np_like.choose(which[0].astype('int'), cases, mode='clip'),
                      self.np_like.choose(which[1].astype('int'), cases, mode='clip'))
            return lb, ub
        return self.np_like.choose(which.astype('int'), cases, mode='wrap')

    ####################################### Arithmetic Operations ############################

    def negative(self, a: Union[NDArrayLike, IntervalLike]) -> Union[NDArray, Interval]:
        if self._is_interval(a):
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

    def subtract(self, a: Union[NDArrayLike, IntervalLike], b: Union[NDArrayLike, IntervalLike]) -> Union[
        NDArray, Interval]:

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

    def multiply(self, a: Union[NDArrayLike, IntervalLike], b: Union[NDArrayLike, IntervalLike]) -> Union[
        NDArray, Interval]:
        return self._arbitrary_bilinear(a, b, self.np_like.multiply, assume_product=True)

    def divide(self, a: Union[NDArrayLike, IntervalLike], b: NDArray) -> Union[NDArray, Interval]:
        b = self.np_like.divide(1, b)
        return self.multiply(a, b)

    def tensordot(self, a: Union[NDArrayLike, IntervalLike], b: Union[NDArrayLike, IntervalLike], axes) -> Union[
        NDArray, Interval]:
        bilinear = partial(self.np_like.tensordot, axes=axes)
        return self._arbitrary_bilinear(a, b, bilinear, assume_product=axes == 0)

    def dot_general(self, lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None):
        (axes, (_, _)) = dimension_numbers
        return self.tensordot(lhs, rhs, axes)

    def _arbitrary_bilinear(self, a: Union[NDArrayLike, IntervalLike], b: Union[NDArrayLike, IntervalLike],
                            bilinear: Callable[[NDArrayLike, NDArrayLike], NDArrayLike],
                            assume_product: bool = False) -> Union[NDArray, Interval]:

        a_is_interval = isinstance(a, tuple)
        b_is_interval = isinstance(b, tuple)

        if not a_is_interval and not b_is_interval:
            return bilinear(a, b)

        return custom_bilinear(a, b, bilinear, assume_product, self.np_like)

    ##################################### Trig/Log/Exp/Power Operations ############################

    def tanh(self, a: Union[NDArrayLike, IntervalLike]) -> Union[NDArray, Interval]:
        if self._is_interval(a):
            return self.np_like.tanh(a[0]), self.np_like.tanh(a[1])
        return self.np_like.tanh(a)

    def sqrt(self, a: Union[NDArrayLike, IntervalLike]) -> Union[NDArray, Interval]:
        if self._is_interval(a):
            return self.np_like.sqrt(a[0]), self.np_like.sqrt(a[1])
        return self.np_like.sqrt(a)

    ##################################### Reduction/Expansion Operations #######################

    def sum(self, a: Union[NDArrayLike, IntervalLike], axes: int = 0):
        if self._is_interval(a):
            return self.np_like.sum(a[0], axes), self.np_like.sum(a[1], axes)
        return self.np_like.sum(a, axes)

    def slice(self, a: Union[NDArrayLike, IntervalLike], *args, **kwargs):
        if self._is_interval(a):
            return utils.slice(self.np_like, a[0], args, kwargs), utils.slice(self.np_like, a[1], args, kwargs)
        return utils.slice(self.np_like, a, args, kwargs)

    def squeeze(self, a: Union[NDArrayLike, IntervalLike], axis=None) -> Union[NDArrayLike, IntervalLike]:
        if self._is_interval(a):
            return self.np_like.squeeze(a[0], axis), self.np_like.squeeze(a[1], axis)
        return self.np_like.squeeze(a, axis)

    def pad(self, a: Union[NDArrayLike, IntervalLike],  pad_width, mode='constant', **kwargs) -> Union[NDArrayLike, IntervalLike]:
        if self._is_interval(a):
            return self.np_like.pad(a[0], pad_width, mode, kwargs), self.np_like.pad(a[1], pad_width, mode, kwargs)
        return self.np_like.pad(a, pad_width, mode, kwargs)

    def broadcast_in_dim(self, operand, *args, **kwargs):
        if self._is_interval(operand):
            return utils.broadcast_in_dim(self.np_like, operand[0], args, kwargs), \
                   utils.broadcast_in_dim(self.np_like, operand[1], args, kwargs)
        return utils.broadcast_in_dim(self.np_like, operand, args, kwargs)

    ##################################### Transformation Operations ########################

    def transpose(self, a: Union[NDArrayLike, IntervalLike], permutation=tuple) -> Union[NDArray, Interval]:
        if self._is_interval(a):
            return self.np_like.transpose(a[0], permutation), self.np_like.transpose(a[1], permutation)
        return self.np_like.transpose(a, permutation)


if __name__ == "__main__":
    import numpy as np
    import jax.numpy as jnp

    a = (np.zeros((1, 3)), np.ones((1, 3)))
    b = (np.zeros((3,)), np.ones((3,)))
    ans = (np.array([0.]), np.array([3.]))
    calculated = IntervalArithmetic(jnp).arbitrary_bilinear(a, b, partial(np.tensordot, axes=1), assume_product=False)
    print(calculated)
