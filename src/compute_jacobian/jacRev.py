from src.site_packages.custom_interpreter.interpreter import Interpreter
from src.model.standaloneNN import inputs, parameters, loss
import jax
interpret = Interpreter()

scalar_x, ival_x = inputs()
params = parameters()
y = loss(scalar_x, params)
print(y)
print("-"*50)

expr = jax.make_jaxpr(loss)(scalar_x, params)
y = interpret.safe_interpret(expr.jaxpr, expr.literals, scalar_x, params)
print(y)

expr = jax.make_jaxpr(jax.jacrev(loss, argnums=(0, 1)))(scalar_x, params)
y = interpret.safe_interpret(expr.jaxpr, expr.literals, scalar_x, params)
print(y)


# prmals_out, vjp_func = jax.vjp(loss, scalar_x, params)
# t1 = vjp_func((1.0, 1.0))
# print("*" * 50)
# expr = jax.make_jaxpr(partial(jax.vjp, loss))(scalar_x, params)
# y = interpret.safe_interpret(expr.jaxpr, expr.literals, (scalar_x, *params))
# print(y)
# print()

# expr = jax.make_jaxpr(jax.jacrev(loss, argnums=(0, 1)))(scalar_x, params)
# y2 = interpret.safe_interpret(expr.jaxpr, expr.literals, (ival_x, *params))
# print(y2)
# print(all(y2[0] < y2[1]))
# print(y2[0][0] < y2[0][1])
# print(y2[1][0] < y2[1][1])
# expr = jax.make_jaxpr(jax.grad(loss2))(scalar_x, params)
# y = interpret.safe_interpret(expr.jaxpr, expr.literals, (scalar_x, *params))[0]
# y = interpret.safe_interpret(expr.jaxpr, expr.literals, (ival_x, *params))[0]
# print(all(y[0] < y[1]))
