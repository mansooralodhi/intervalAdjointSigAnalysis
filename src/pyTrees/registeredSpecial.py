
import jax
from jax.tree_util import register_pytree_node
from special import Special


class RegisteredSpecial(Special):
  def __repr__(self):
    return "RegisteredSpecial(x={}, y={})".format(self.x, self.y)

  def __array__(self):
      return jax.numpy.array([self.x, self.y])

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
  children = (v.x, v.y)
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
  return RegisteredSpecial(*children)

# Global registration
register_pytree_node(
    RegisteredSpecial,
    special_flatten,    # Instruct JAX what are the children nodes.
    special_unflatten   # Instruct JAX how to pack back into a `RegisteredSpecial`.
)

if __name__ == "__main__":
    # print(jax.tree_map(lambda x: x + 1,
    #     [
    #       RegisteredSpecial(0, 1),
    #       RegisteredSpecial(2, 4),
    #     ]))
    special1 = RegisteredSpecial(1, 5)
    special2 = RegisteredSpecial(2, 3)
    print(jax.numpy.add(special1, special2))