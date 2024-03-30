
import jax
from jax.tree_util import register_pytree_node
from src.thesis_prototype_v2.intervals import Interval


class RegisteredInterval(Interval):
  def __repr__(self):
    return "RegisteredSpecial(x={}, y={})".format(self.lower_bound, self.upper_bound)

def special_flatten(v):
  """Specifies a flattening recipe.

  Params:
    v: The value of the registered type to flatten.
  Returns:
    A pair of an iterable with the children to be flattened recursively,
    and some opaque auxiliary data to pass back to the unflattening recipe.
    The auxiliary data is stored in the treedef for use during unflattening.
    The auxiliary data could be used, for example, for dictionary keys.
  """
  children = (v.lower_bound, v.upper_bound)
  aux_data = None
  return (children, aux_data)

def special_unflatten(aux_data, children):
  """Specifies an unflattening recipe.

  Params:
    aux_data: The opaque data that was specified during flattening of the
      current tree definition.
    children: The unflattened children

  Returns:
    A reconstructed object of the registered type, using the specified
    children and auxiliary data.
  """
  return RegisteredInterval(*children)
  # del aux_data
  # obj = object.__new__(Interval)
  # obj.lower_bound = 1.0
  # obj.upper_bound = 2.0
  # return obj

# Global registration
register_pytree_node(
    RegisteredInterval,
    special_flatten,    # Instruct JAX what are the children nodes.
    special_unflatten   # Instruct JAX how to pack back into a `RegisteredSpecial`.
)

if __name__ == "__main__":
    print(jax.tree_map(lambda x: x + 1,
        [
          RegisteredInterval(0, 1),
          RegisteredInterval(2, 4),
        ]))