


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

expr = jax.make_jaxpr(jax.jacfwd(loss))(scalar_x, params)
y = interpret.safe_interpret(expr.jaxpr, expr.literals, scalar_x, params)
print(y)

