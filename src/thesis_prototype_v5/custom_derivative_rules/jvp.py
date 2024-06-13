
import jax.numpy as jnp
from jax import custom_jvp, jvp, grad

"""
1.  If we decorate a function with @custom_jvp than its compulsory to define its jvp rule.
2.  The custom rule can be implemented in either of the two methods below.
3.  The custom rule would be automatically called when using the built-in method jax.jvp.
4.  An additional feature of jax.jvp is that it can seed more than one input at a time 
    and sum the output tangents of those seeded inputs.
"""

@custom_jvp
def f(x, y):
    return jnp.sin(x) * y

############################### Method-1 (for defining jvp) #####################
# jvp_wrt_x = lambda x_dot, primal_out, x, y: jnp.cos(x) * x_dot * y
# jvp_wrt_y = lambda y_dot, primal_out, x, y: jnp.sin(x) * y_dot
# f.defjvps(jvp_wrt_x, jvp_wrt_y)  # register the custom jvp rule.


############################### Method-2 (for defining jvp) #####################
@f.defjvp
def f_jvp(primals, tangents):
  x, y = primals
  x_dot, y_dot = tangents
  primal_out = f(x, y)
  tangent_out = jnp.cos(x) * x_dot * y + jnp.sin(x) * y_dot
  return primal_out, tangent_out


if __name__ == "__main__":
    print("primal computed =  ", f(2., 3.))
    primal, tangent = jvp(f, (2., 3.), (0., 1.))
    print("custom primal computed = ", primal)
    print("tangent computed =  ", grad(f, argnums=1)(2., 3.))
    print("custom tangent computed =  ", tangent)
