

from typing import Union
from abc import abstractmethod
from typing_extensions import Protocol, runtime_checkable
from src.thesis_prototype_v5.custom_interpreter.ivalOps.ndarray import NDArray

"""
Key Features:
    -   interval: anything that is enclosed in tuple.
    -   non-interval: anything that is not enclosed in tuple.
"""

# todo: understand how to check the types of below datatypes

Interval = tuple['NDArray', 'NDArray']
NDArrayLike = Union['NDArray', int, float]
IntervalLike = tuple[NDArrayLike, NDArrayLike]

@runtime_checkable
class NumpyLike(Protocol):
    """
    Could be numpy or jax.numpy !
    """
    @abstractmethod
    def add(self, a: NDArrayLike, b: NDArrayLike) -> NDArrayLike: pass

    @abstractmethod
    def subtract(self, a: NDArrayLike, b: NDArrayLike) -> NDArray: pass

    @abstractmethod
    def amax(self, *args, **kwargs): pass

    @abstractmethod
    def amin(self, *args, **kwargs): pass

    @abstractmethod
    def average(self, *args, **kwargs): pass

    @abstractmethod
    def broadcast_to(self,  *args, **kwargs): pass

    @abstractmethod
    def concatenate(self, *args, **kwargs): pass

    @abstractmethod
    def cos(self, *args, **kwargs): pass

    @abstractmethod
    def sin(self, *args, **kwargs): pass

    @abstractmethod
    def tan(self, *args, **kwargs): pass

    @abstractmethod
    def divide(self, a: NDArrayLike, b: NDArrayLike) -> NDArray: pass

    @abstractmethod
    def multiply(self, a: NDArrayLike, b: NDArrayLike) -> NDArray: pass

    @abstractmethod
    def maximum(self, a: NDArrayLike, b: NDArrayLike) -> NDArray: pass

    @abstractmethod
    def minimum(self, a: NDArrayLike, b: NDArrayLike) -> NDArray: pass

    @abstractmethod
    def sum(self, a: NDArrayLike, **kwargs) -> NDArray: pass

    @abstractmethod
    def dot(self, *args, **kwargs): pass

    @abstractmethod
    def einsum(self, *args, **kwargs): pass

    @abstractmethod
    def exp(self, *args, **kwargs): pass

    @abstractmethod
    def inner(self, *args, **kwargs): pass

    @abstractmethod
    def log(self, x: NDArrayLike)-> NDArray: pass

    @abstractmethod
    def matmul(self, *args, **kwargs): pass

    @abstractmethod
    def max(self, *args, **kwargs): pass

    @abstractmethod
    def min(self, *args, **kwargs): pass

    @abstractmethod
    def mean(self, *args, **kwargs): pass

    @abstractmethod
    def ndim(self, a: NDArrayLike) -> int: pass

    @abstractmethod
    def power(self, a, exponent) -> NDArray: pass

    @abstractmethod
    def prod(self, *args, **kwargs): pass

    @abstractmethod
    def reshape(self, a: NDArrayLike, shape) -> NDArray: pass

    @abstractmethod
    def shape(self, a: NDArrayLike): pass

    @abstractmethod
    def sqrt(self, *args, **kwargs): pass

    @abstractmethod
    def square(self, *args, **kwargs): pass

    @abstractmethod
    def trace(self, *args, **kwargs): pass

    @abstractmethod
    def true_divide(self, *args, **kwargs): pass

    @abstractmethod
    def transpose(self, a, permutation): pass

    @abstractmethod
    def tanh(self, a):
        pass


if __name__ == '__main__':
    import numpy as np
    x = np.array(1.0)
    xIval = (x, x)
    isNdArray = isinstance(x, NDArray)
