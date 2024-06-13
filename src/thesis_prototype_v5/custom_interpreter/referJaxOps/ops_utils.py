

import itertools
import numpy as np
from jax._src import dtypes


def broadcast(operand, out_dim_size, broadcast_dimensions):
    """
    Source: https://github.com/google/jax/blob/f5cc272615ce2795f9133e63b7b535ec5ada7e52/jax/_src/lax_reference.py#L245
    """
    in_reshape = np.ones(len(out_dim_size), dtype=np.int32)
    for i, bd in enumerate(broadcast_dimensions):
        in_reshape[bd] = operand.shape[i]
    return np.broadcast_to(np.reshape(operand, in_reshape), out_dim_size)

def contract_dimensions(lhs, rhs, dimension_numbers):
    """
    Source: https://github.com/google/jax/blob/4d4151db8e76beedba9df268293653b35ea68e28/jax/_src/lax_reference.py#L209
    """
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
    new_id = itertools.count()
    lhs_axis_ids = [next(new_id) for _ in lhs.shape]
    rhs_axis_ids = [next(new_id) for _ in rhs.shape]
    lhs_out_axis_ids = lhs_axis_ids[:]
    rhs_out_axis_ids = rhs_axis_ids[:]

    for lhs_axis, rhs_axis in zip(lhs_contracting, rhs_contracting):
        shared_id = next(new_id)
        lhs_axis_ids[lhs_axis] = shared_id
        rhs_axis_ids[rhs_axis] = shared_id
        lhs_out_axis_ids[lhs_axis] = None
        rhs_out_axis_ids[rhs_axis] = None

    batch_ids = []
    for lhs_axis, rhs_axis in zip(lhs_batch, rhs_batch):
        shared_id = next(new_id)
        lhs_axis_ids[lhs_axis] = shared_id
        rhs_axis_ids[rhs_axis] = shared_id
        lhs_out_axis_ids[lhs_axis] = None
        rhs_out_axis_ids[rhs_axis] = None
        batch_ids.append(shared_id)

    not_none = lambda x: x is not None
    out_axis_ids = filter(not_none,
                          batch_ids + lhs_out_axis_ids + rhs_out_axis_ids)
    assert lhs.dtype == rhs.dtype
    dtype = np.float32 if lhs.dtype == dtypes.bfloat16 else None
    out = np.einsum(lhs, lhs_axis_ids, rhs, rhs_axis_ids, out_axis_ids,
                    dtype=dtype)
    return out.astype(dtypes.bfloat16) if lhs.dtype == dtypes.bfloat16 else out