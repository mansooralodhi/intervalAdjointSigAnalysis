
from jax import core
from jax.tree_util import tree_flatten, tree_unflatten, tree_map


def show_example(mytree, is_spvalue):
  flat, tree = tree_flatten(mytree, is_leaf=is_spvalue)
  unflattened = tree_unflatten(tree, flat)
  print(f"{mytree=}\n  {flat=}\n  {tree=}\n  {unflattened=}")


def ivals_to_avals(interval_values, is_intvalue):
  """Convert a pytree of Tree to an equivalent pytree of abstract values."""

  def ival_to_aval(ival):
    # data = spenv.data(ival)
    # return core.ShapedArray(ival.shape, data.dtype, data.aval.weak_type)
    return core.ShapedArray(ival.shape, float, True)
  return tree_map(ival_to_aval, interval_values, is_leaf=is_intvalue)
