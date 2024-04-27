


from src.makeModel.modelRuntime import ModelRuntime
from src.makeModel.modelJaxpr import ModelJaxpr
from src.makeModel.modelGrads import ModelGrad
from src.customInterpreter.utils import merge_args
from src.customInterpreter.interpret import safe_interpret

modelRuntime = ModelRuntime()
modelJaxpr = ModelJaxpr(modelRuntime)
modelGrads = ModelGrad(modelRuntime)

x = modelRuntime.sampleX
params = modelRuntime.model_params


###################################### Check forward-pass #####################
loss = modelRuntime.loss(x, params)
expr = modelJaxpr.primal_jaxpr(x, params)
inp = merge_args(x, params)
est_loss = safe_interpret(expr.jaxpr, expr.literals, inp)[0]
print(f"Actual Loss: {loss}, Interpreted Loss: {est_loss}")

##################################### Check reverse-pass #####################

# NB: the resulting structure would be different with different value of wrt_arg.
grad = modelGrads.grad(x, params, wrt_arg=0)
expr = modelJaxpr.adjoint_jaxpr(x, params, wrt_arg=0)
inp = merge_args(x, params)
est_grad = safe_interpret(expr.jaxpr, expr.literals, inp)[0]
print(all(grad == est_grad))