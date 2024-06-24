import builtins

_slice = builtins.slice
import numpy as np


def slice(np_like, operand, start_indices, limit_indices, strides=None):  # pylint: disable=redefined-builtin
    if strides is None:
        strides = np_like.ones(len(start_indices)).astype(int)
    slices = tuple(map(_slice, start_indices, limit_indices, strides))
    return operand[slices]


def broadcast_in_dim(np_like, operand, shape, broadcast_dimensions):
    in_reshape = np.ones(len(shape),
                         dtype=np_like.int32)  # NB: if this is changed to jnp then we get trace error __index__
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


def lax2numpy_pad(np_like, array, padding_config, padding_value):
    pad_width = [(low, high) for low, high, _ in padding_config]
    padded_array = np_like.pad(array, pad_width, mode='constant', constant_values=padding_value)

    for dim, (low, high, interior) in enumerate(padding_config):
        if interior > 0:
            # Interleave zeros in the padded_array along the specified dimension
            shape = list(padded_array.shape)
            new_shape = shape[:dim] + [shape[dim] * (interior + 1) - interior] + shape[dim + 1:]
            interleaved_array = np_like.zeros(new_shape, dtype=padded_array.dtype)

            indices = [_slice(None)] * len(shape)
            for i in range(shape[dim]):
                indices[dim] = _slice(i * (interior + 1), i * (interior + 1) + 1)
                interleaved_array[tuple(indices)] = padded_array[
                    tuple([_slice(None) if d != dim else _slice(i, i + 1) for d in range(len(shape))])]
            padded_array = interleaved_array

    return padded_array
