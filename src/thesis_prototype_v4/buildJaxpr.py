import jax.numpy
import numpy as np
from jax import core, tree_map
from jax.core import Jaxpr
from typing import Tuple, List, Union
from jax.tree_util import tree_flatten
from jax._src.api_util import flatten_fun_nokwargs
from jax import linear_util as lu
from jax.interpreters import partial_eval as pe


class BuildJaxpr:

    customDtype = []
    is_leaf = lambda x: type(x) in BuildJaxpr.customDtype

    @classmethod
    def to_avals(cls, args: Union[Tuple, List]) -> any:
      def to_aval(leaf):
        if isinstance(leaf, float):
          leaf = jax.numpy.asarray(leaf)
          shape, dtype = leaf.shape, leaf.dtype
        elif isinstance(leaf, np.ndarray):
          shape, dtype = leaf.shape, jax.numpy.float32
        elif isinstance(leaf, jax.numpy.ndarray):
          shape, dtype = leaf.shape, leaf.dtype
        elif type(leaf) in cls.customDtype:
          shape, dtype = (), jax.numpy.float32
        else:
          raise Exception(f"Leaf of Type {type(leaf)} Not Supported !")
        return core.ShapedArray(shape, dtype)
      return tree_map(to_aval, args, is_leaf=cls.is_leaf)

    @classmethod
    def build(cls, f, *args: Union[Tuple, List]) -> Tuple[Jaxpr, Union[Tuple, List]]:
        args_flat, in_tree = tree_flatten(args, is_leaf=cls.is_leaf)
        in_avals_flat = cls.to_avals(args_flat)
        wrapped_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(f, {}), in_tree)
        # note: abstract values are used to trace the and the computational graph leading to jaxpr.
        jaxpr, out_avals_flat, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, in_avals_flat)
        return jaxpr, consts

