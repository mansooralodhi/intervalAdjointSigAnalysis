from jax.tree_util import register_pytree_node
import jax.numpy as jnp
import jax

class MyTree:
  def __init__(self, a, b):
    if not (type(a) is object or a is None or isinstance(a, MyTree)):
      a = jnp.asarray(a)
    if not (type(b) is object or b is None or isinstance(b, MyTree)):
      b = jnp.asarray(b)
    self.a = a
    self.b = b

register_pytree_node(MyTree, lambda tree: ((tree.a,tree.b), None),
    lambda _, args: MyTree(*args))

tree = MyTree(jnp.arange(5.0), jnp.arange(5.0))
# tree = MyTree(jnp.arange(5.0))


print(jnp.add(tree, tree))
# jax.vmap(lambda x: x)(tree)      # Error because object() is passed to MyTree.
# jax.jacobian(lambda x: x)(tree)  # Error because MyTree(...) is passed to MyTree