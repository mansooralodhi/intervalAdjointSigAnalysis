
from jax.tree_util import tree_structure


def validate_scalar(J_out, K_out):
    print(tree_structure(J_out))
    print(tree_structure(K_out))