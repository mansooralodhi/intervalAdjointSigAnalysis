

from abc import abstractmethod
from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class NDArray(Protocol):

    ############################### magic methods ##############################

    @abstractmethod
    def __add__(self, other) -> 'NDArray': pass

    @abstractmethod
    def __radd__(self, other) -> 'NDArray': pass

    @abstractmethod
    def __sub__(self, other) -> 'NDArray': pass

    @abstractmethod
    def __rsub__(self, other) -> 'NDArray': pass

    @abstractmethod
    def __mul__(self, other) -> 'NDArray': pass

    @abstractmethod
    def __rmul__(self, other) -> 'NDArray': pass

    @abstractmethod
    def __truediv__(self, other) -> 'NDArray': pass

    @abstractmethod
    def __rtruediv__(self, other) -> 'NDArray': pass

    @abstractmethod
    def __matmul__(self, other) -> 'NDArray': pass

    @abstractmethod
    def __rmatmul__(self, other) -> 'NDArray': pass

    @abstractmethod
    def __pow__(self, power, modulo=None) -> 'NDArray': pass

    @abstractmethod
    def __rpow__(self, power, modulo=None) -> 'NDArray': pass

    ################################# property ################################

    @property
    def ndim(self) -> int: raise NotImplementedError(self.__class__)

    @property
    def shape(self) -> tuple: raise NotImplementedError(self.__class__)

    ############################### public methods ##############################

    @abstractmethod
    def max(self, *args, **kwargs) -> 'NDArray': pass

    @abstractmethod
    def min(self, *args, **kwargs) -> 'NDArray': pass

    @abstractmethod
    def mean(self, *args, **kwargs) -> 'NDArray': pass

    @abstractmethod
    def reshape(self, *args, **kwargs) -> 'NDArray': pass

    @abstractmethod
    def transpose(self,*axes) -> 'NDArray': pass


if __name__ == "__main__":
    import torch
    import numpy as np
    import jax.numpy as jnp

    print(isinstance(int, NDArray))             # False
    print(isinstance(float, NDArray))           # False
    print(isinstance(np.ndarray, NDArray))      # True
    print(isinstance(jnp.ndarray, NDArray))     # True
    print(isinstance(torch.tensor, NDArray))    # True
    print(isinstance(torch.Tensor, NDArray))    # True




