

from src.thesis_prototype_v5.model.denseNN import DenseNN
from src.thesis_prototype_v5.custom_interpreter.interpreter import Interpreter

import jax
import jax.numpy as jnp
from jax import grad, random as rnd

model = DenseNN()
X = rnd.normal(rnd.key(0), shape=(264,))
y = model.calculate_y(X)
print(f"J_prim = {y}")

interpret = Interpreter()

jaxExpr = jax.make_jaxpr(model.calculate_y)(X)
y = interpret.safe_interpret(jaxExpr.jaxpr, jaxExpr.literals, [X])[0]
print(f"K_prim = {y}")

X_lb, X_ub = X-0.05, X+0.05
y_ival = interpret.safe_interpret(jaxExpr.jaxpr, jaxExpr.literals, [(X_lb, X_ub)])[0]
print(f"K_prim ival = {y_ival}")

y_grad = grad(model.calculate_y)(X)
print(f"J_adj len = {len(y_grad)}")

y_grad_fn = grad(model.calculate_y)
jaxExpr = jax.make_jaxpr(y_grad_fn)(X)
y = interpret.safe_interpret(jaxExpr.jaxpr, jaxExpr.literals, [X])[0]
print(f"K_adj len = {len(y)}")

print(f"all (J_adj == K_adj) = {all(y_grad == y)}")

y_ival = interpret.safe_interpret(jaxExpr.jaxpr, jaxExpr.literals, [(X_lb, X_ub)])[0]
print(f"K_adj ival all(lb<ub) = {all(y_ival[0] < y_ival[1])}")
