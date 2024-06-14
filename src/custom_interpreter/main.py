

import jax
from src.model.loader import ModelLoader
from src.model.runtime import ModelRuntime
from src.custom_interpreter.interpret_v0 import safe_interpret

modelLoader = ModelLoader()
modelRuntime = ModelRuntime(modelLoader.model)

x = modelLoader.sampleX
params = modelLoader.model_params


###################################### Check forward-pass #####################
loss = modelRuntime.loss(x, params)
expr = modelRuntime.primal_jaxpr(x, params)
flatParams, _ = jax.tree_flatten(params)
flatParams.insert(0, x)
est_loss = safe_interpret(expr.jaxpr, expr.literals, flatParams)[0]
print(f"Actual Loss: {loss}, Interpreted Loss: {est_loss}")

##################################### Check reverse-pass #####################

# NB: the resulting structure would be different with different value of wrt_arg.
grad = modelRuntime.grad(x, params, wrt_arg=0)
expr = modelRuntime.adjoint_jaxpr(x, params, wrt_arg=0)
flatParams, _ = jax.tree_flatten(params)
flatParams.insert(0, x)
est_grad = safe_interpret(expr.jaxpr, expr.literals, flatParams)[0]
print(all(grad == est_grad))