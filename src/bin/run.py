import jax
from src.model.runtime import ModelRuntime
from src.site_packages.custom_interpreter import safe_interpreter
from functools import wraps

modelRuntime = ModelRuntime()

x = modelRuntime.sampleX
params = modelRuntime.model_params

featuresInterval = modelRuntime.feature_intervals
featureIval = (featuresInterval[0], featuresInterval[1])  # it's necessary to pass tuple


##########################################   Test Primals #####################################
#
# loss = modelRuntime.loss(x, params)
# expr = modelRuntime.primal_jaxpr(x, params)
# print(expr)
# print("-"*50)
# flatParams, _ = jax.tree_flatten(params)
# flatParams.insert(0, x)
# x_est_loss = safe_interpreter(expr.jaxpr, expr.literals, flatParams)[0]
# flatParams, _ = jax.tree_flatten(params)
# flatParams.insert(0, featureIval)
# ival_est_loss = safe_interpreter(expr.jaxpr, expr.literals, flatParams)[0]
#
# print(f"Actual Loss: {loss}\n"
#       f"Interpreted Loss: {x_est_loss}\n"
#       f"Interpreted Interval Loss: {ival_est_loss}\n")

##########################################   Test Primals #####################################

# adjoints = modelRuntime.grad(x, params, wrt_arg=(0))
# expr = modelRuntime.adjoint_jaxpr(x, params, wrt_arg=(0))
# print(expr)
# flatParams, _ = jax.tree_flatten(params)
# flatParams.insert(0, x)
# x_est_adjoint = safe_interpreter(expr.jaxpr, expr.literals, flatParams)[0]
# flatParams, _ = jax.tree_flatten(params)
# flatParams.insert(0, featureIval)
# ival_est_adjoint = safe_interpreter(expr.jaxpr, expr.literals, flatParams)[0]
#
# adjoint_bounded = jax.numpy.dot((x_est_adjoint - ival_est_adjoint[0]),(x_est_adjoint - ival_est_adjoint[1])) < 0
# print(f" Are adjoints bounded ? {adjoint_bounded}")
# print()
# print(f"Adjoints == Interpreted Adjoints:  {all(adjoints == x_est_adjoint)}\n"
#       f"Len of Interpreted Adjoints: {len(ival_est_adjoint)}\n"
#       f"Interpreted Adjoint Lb < UB: {all(ival_est_adjoint[0]<ival_est_adjoint[1])}")

##########################################   Jaxpr of Jaxpr  #####################################
f = modelRuntime.loss

def concatenate_args(x, params):
    flatParams, _ = jax.tree.flatten(params)
    flatParams.insert(0, x)
    return flatParams

def concatenate_args_v1(x, params):
    flatParams, _ = jax.tree.flatten(params)
    flatParams.insert(0, x)
    flatParams.insert(len(flatParams), x)
    flatParams.insert(len(flatParams), x)
    return flatParams

def interval(fun):
  @wraps(fun)
  def wrapped(x, params, intervals=None):
    closed_jaxpr = jax.make_jaxpr(fun)(x, params)
    args = concatenate_args(intervals, params)
    out = safe_interpreter(closed_jaxpr.jaxpr, closed_jaxpr.literals, args)
    return out
  return wrapped

ival_f = interval(jax.grad(f))
expr = jax.make_jaxpr(ival_f)(x, params, intervals=featureIval)
print(expr)
out = safe_interpreter(expr.jaxpr, expr.literals, concatenate_args_v1(featureIval, params))
# print(out)

