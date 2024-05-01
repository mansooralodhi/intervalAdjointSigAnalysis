
import jax
from src.model.runtime import ModelRuntime
from src.interpreter.interpret import safe_interpret

modelRuntime = ModelRuntime()

x = modelRuntime.sampleX
params = modelRuntime.model_params

featuresInterval = modelRuntime.feature_intervals
featureIval = (featuresInterval[1], featuresInterval[0]) # it's necessary to pass tuple

loss = modelRuntime.loss(x, params)
expr = modelRuntime.primal_jaxpr(x, params)
flatParams, _ = jax.tree_flatten(params)
flatParams.insert(0, x)
x_est_loss = safe_interpret(expr.jaxpr, expr.literals, flatParams)[0]
flatParams, _ = jax.tree_flatten(params)
flatParams.insert(0, featureIval)
ival_est_loss = safe_interpret(expr.jaxpr, expr.literals, flatParams)[0]
print(loss)
print(ival_est_loss)

adjoints = modelRuntime.grad(x, params, wrt_arg=(0))
expr = modelRuntime.adjoint_jaxpr(x, params, wrt_arg=(0))
flatParams, _ = jax.tree_flatten(params)
flatParams.insert(0, x)
x_est_adjoint = safe_interpret(expr.jaxpr, expr.literals, flatParams)
flatParams, _ = jax.tree_flatten(params)
flatParams.insert(0, featureIval)
ival_est_adjoint = safe_interpret(expr.jaxpr, expr.literals, flatParams)
print(ival_est_adjoint[0])
