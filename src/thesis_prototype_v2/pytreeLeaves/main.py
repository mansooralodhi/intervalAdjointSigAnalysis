
import numpy as np
from tree import Tree
from utils import show_example, ivals_to_avals
from jax.tree_util import tree_flatten, tree_unflatten

f = lambda mat, vec: np.matmul(mat, vec)
is_interval = lambda arg: isinstance(arg, Tree)

ival = Tree(1., 2.)
vec = np.asarray([ival, ival, ival])
mat = np.asarray([[ival, ival, ival],
                  [ival, ival, ival]])
print(vec)
print(vec.shape)
print(mat.shape)
args = [mat, vec, [2, 4, 5], [ival, ival]]

# while tree_flatten, branches are created at collections/container types like list/tuple/dict
# not at container types like numpy, customClass,

values_flat, in_tree = tree_flatten(args, )
print(len(values_flat))
print(in_tree)

# in_avals_flat = spvalues_to_avals(spenv, spvalues_flat)
# wrapped_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(f, params), in_tree)
# jaxpr, out_avals_flat, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, in_avals_flat)

import jax
import jax.numpy as jnp
f = lambda x, y: y * x
vec = np.asarray([[1., 1., 1.]])
print(f(vec, vec))
print(jax.vjp(f, vec, vec))
