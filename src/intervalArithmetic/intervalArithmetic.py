


from functools import partial
from typing import Union, Callable
from src.intervalArithmetic.intervalArithmetic_utils import *
from src.intervalArithmetic.numpyLike import (NDArray, NDArrayLike, Interval, IntervalLike, NumpyLike)

class IntervalArithmetic():

    def __init__(self, np_like: NumpyLike):
        self.np_like = np_like

    ########################### Transformation Operations ####################

    def as_interval(self, a: IntervalLike) -> Interval:
        return self.np_like.asarray(a[0]), self.np_like.asarray(a[1])

    def as_interval_or_ndarray(self, a: Union[NDArrayLike, IntervalLike]) -> Union[NDArray, Interval]:
        if isinstance(a, tuple):
            return self.as_interval(a)
        return self.np_like.asarray(a)

    ############################# Attribute Operations #######################

    def shape(self, a: Union[NDArrayLike, IntervalLike]) -> tuple:
        if isinstance(a, tuple):
            return self.np_like.shape(a[0])
        return self.np_like.shape(a)

    def ndim(self, a: Union[NDArrayLike, IntervalLike]) -> int:
        if isinstance(a, tuple):
            return self.np_like.ndim(a[0])
        return self.np_like.ndim(a)

    ############################## Arithmetic Operations ######################

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

    def tensordot(self, a: Union[NDArrayLike, IntervalLike], b: Union[NDArrayLike, IntervalLike], axes) -> Union[NDArray, Interval]:
        bilinear = partial(self.np_like.tensordot, axes=axes)
        return self._arbitrary_bilinear(a, b, bilinear, assume_product = axes==0)

    # todo
    def outer_product(self, a: Union[NDArrayLike, IntervalLike], b: Union[NDArrayLike, IntervalLike], batch_dims: int = 0) -> Union[NDArray, Interval]:
        """Interval variant of _ndarray_outer_product()."""
        pass

    # todo
    def outer_power(self, a: Union[NDArrayLike, IntervalLike], exponent: int, batch_dims: int = 0) -> Union[NDArray, Interval]:
        """Returns a repeated outer product."""
        pass

    # todo
    def power(self, a: Union[NDArrayLike, IntervalLike], exponent: int, batch_dims: int = 0) -> Union[NDArray, Interval]:
        """Returns a**exponent (element-wise)."""
        pass

    def _arbitrary_bilinear(self, a: Union[NDArrayLike, IntervalLike], b: Union[NDArrayLike, IntervalLike],
                            bilinear: Callable[[NDArrayLike, NDArrayLike], NDArrayLike], assume_product: bool=False) -> Union[NDArray, Interval]:

        a_is_interval = isinstance(a, tuple)
        b_is_interval = isinstance(b, tuple)

        if not a_is_interval and not b_is_interval:
            return bilinear(a, b)

        return custom_bilinear(a, b, bilinear, assume_product, self.np_like)

    ########################### Reduction/Expansion Operations ################



if __name__ == "__main__":

    import numpy as np
    import jax.numpy as jnp

    a = (np.zeros((1, 3)), np.ones((1, 3)))
    b = (np.zeros((3,)), np.ones((3,)))
    ans =  (np.array([0.]), np.array([3.]))
    calculated = IntervalArithmetic(jnp).arbitrary_bilinear(a, b, partial(np.tensordot, axes=1), assume_product=False)
    print(calculated)
















