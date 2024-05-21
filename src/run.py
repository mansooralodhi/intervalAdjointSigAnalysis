
import jax
from src.model.runtime import ModelRuntime
from src.interpreter.interpreter import safe_interpret

modelRuntime = ModelRuntime()

x = modelRuntime.sampleX
params = modelRuntime.model_params

featuresInterval = modelRuntime.feature_intervals
featureIval = (featuresInterval[0], featuresInterval[1]) # it's necessary to pass tuple

##########################################   Test Primals #####################################
#
# loss = modelRuntime.loss(x, params)
# expr = modelRuntime.primal_jaxpr(x, params)
# print(expr)
# print("-"*50)
# flatParams, _ = jax.tree_flatten(params)
# flatParams.insert(0, x)
# x_est_loss = safe_interpret(expr.jaxpr, expr.literals, flatParams)[0]
# flatParams, _ = jax.tree_flatten(params)
# flatParams.insert(0, featureIval)
# ival_est_loss = safe_interpret(expr.jaxpr, expr.literals, flatParams)[0]
#
# print(f"Actual Loss: {loss}\n"
#       f"Interpreted Loss: {x_est_loss}\n"
#       f"Interpreted Interval Loss: {ival_est_loss}\n")
##########################################   Test Primals #####################################

adjoints = modelRuntime.grad(x, params, wrt_arg=(0))
expr = modelRuntime.adjoint_jaxpr(x, params, wrt_arg=(0))
print(expr)
flatParams, _ = jax.tree_flatten(params)
flatParams.insert(0, x)
x_est_adjoint = safe_interpret(expr.jaxpr, expr.literals, flatParams)[0]
flatParams, _ = jax.tree_flatten(params)
flatParams.insert(0, featureIval)
ival_est_adjoint = safe_interpret(expr.jaxpr, expr.literals, flatParams)[0]

adjoint_bounded = jax.numpy.dot((x_est_adjoint - ival_est_adjoint[0]),(x_est_adjoint - ival_est_adjoint[1])) < 0
print(f" Are adjoints bounded ? {adjoint_bounded}")
print()
print(f"Adjoints == Interpreted Adjoints:  {all(adjoints == x_est_adjoint)}\n"
      f"Len of Interpreted Adjoints: {len(ival_est_adjoint)}\n"
      f"Interpreted Adjoint Lb < UB: {all(ival_est_adjoint[0]<ival_est_adjoint[1])}")
