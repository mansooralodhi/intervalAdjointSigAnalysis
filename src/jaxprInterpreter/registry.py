

import numpy as np
import jax.numpy as jnp
from jax import lax

inverse_registry = dict()
interval_registry = dict()

inverse_registry[lax.exp_p] = jnp.log
inverse_registry[lax.tanh_p] = jnp.arctanh


interval_registry[lax.tanh_p] = np.tanh
interval_registry[lax.exp_p] = np.exp
interval_registry[lax.add_p] = np.add
interval_registry[lax.dot_general_p] = np.matmul





