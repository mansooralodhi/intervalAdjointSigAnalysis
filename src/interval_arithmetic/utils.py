import builtins
_slice = builtins.slice
import numpy as np

def slice(np_like, operand, start_indices, limit_indices, strides=None):  # pylint: disable=redefined-builtin
  if strides is None:
    strides = np_like.ones(len(start_indices)).astype(int)
  slices = tuple(map(_slice, start_indices, limit_indices, strides))
  return operand[slices]

def broadcast_in_dim(np_like, operand, shape, broadcast_dimensions):
  in_reshape = np.ones(len(shape), dtype=np_like.int32)  # NB: if this is changed to jnp then we get trace error __index__
  for i, bd in enumerate(broadcast_dimensions):
    in_reshape[bd] = operand.shape[i]
  return np_like.broadcast_to(np_like.reshape(operand, in_reshape), shape)


def iota(np_like, dimension, dtype, shape):
  """
  Mimic the behavior of jax.lax.iota in NumPy.

  Parameters:
  dimension (int): The axis along which the indices will be generated.
  dtype (data-type): The data type of the output array.
  shape (tuple): The shape of the output array.

  Returns:
  np.ndarray: A NumPy array with the specified shape and indices generated along the specified dimension.
  """
  result = np_like.zeros(shape, dtype=dtype)
  indices = np_like.arange(shape[dimension], dtype=dtype)
  result = np_like.moveaxis(result, dimension, -1)
  result += indices
  result = np_like.moveaxis(result, -1, dimension)
  return result

