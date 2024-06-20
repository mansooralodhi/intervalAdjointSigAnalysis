import builtins
_slice = builtins.slice

def slice(np_like, operand, start_indices, limit_indices, strides=None):  # pylint: disable=redefined-builtin
  if strides is None:
    strides = np_like.ones(len(start_indices)).astype(int)
  slices = tuple(map(_slice, start_indices, limit_indices, strides))
  return operand[slices]

def broadcast_in_dim(np_like, operand, shape, broadcast_dimensions):
  in_reshape = np_like.ones(len(shape), dtype=np_like.int32)  # NB: if this is changed to jnp then we get trace error __index__
  for i, bd in enumerate(broadcast_dimensions):
    in_reshape[bd] = operand.shape[i]
  return np_like.broadcast_to(np_like.reshape(operand, in_reshape), shape)
