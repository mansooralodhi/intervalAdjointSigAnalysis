import jax
import jax.numpy as jnp

def examine_jaxpr(closed_jaxpr):
  jaxpr = closed_jaxpr.jaxpr
  print("jaxpr invars:", jaxpr.invars)
  print("jaxpr outvars:", jaxpr.outvars)
  print("jaxpr constvars:", jaxpr.constvars)
  for i, eqn in enumerate(jaxpr.eqns):
    print(f"equation {i}: \n\tinvars:{eqn.invars}, eqn.primitive: {eqn.primitive}, "
          f"eqn.outvars: {eqn.outvars}, eqn.params: {eqn.params}")
  print()
  print("jaxpr:", jaxpr)


def foo(x):
  return x + 1

def bar(w, b, x):
  return jnp.dot(w, x) + b + jnp.ones(5), x


if __name__ == "__main__":
  print("foo")
  print("=====")
  examine_jaxpr(jax.make_jaxpr(foo)(5))

  print()

  print("bar")
  print("=====")
  examine_jaxpr(jax.make_jaxpr(bar)(jnp.ones((5, 10)), jnp.ones(5), jnp.ones(10)))