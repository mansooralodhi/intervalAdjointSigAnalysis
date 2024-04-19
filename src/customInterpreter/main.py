


from src.makeModel.modelRuntime import ModelRuntime
from src.makeModel.modelJaxpr import ModelJaxpr
from src.makeModel.modelGrads import ModelGrad
from src.customInterpreter.interpret import safe_interpret


modelRuntime = ModelRuntime()
modelJaxpr = ModelJaxpr(modelRuntime)
modelGrads = ModelGrad(modelRuntime)

x = modelRuntime.sampleX
params = modelRuntime.model_params


###################################### Check forward-pass #####################
loss = modelRuntime.loss(x, params)
expr = modelJaxpr.forward_jaxpr_wrt_inputs(x)
estimated_loss = safe_interpret(expr.jaxpr, expr.literals, x)
print(f"Actual Loss: {loss}\n"
      f"Interpreted Loss: {estimated_loss[0]}\n")

##################################### Check reverse-pass #####################

grad = modelGrads.grad_wrt_inputs()
expr = modelJaxpr.grad_jaxpr_wrt_inputs()
estimated_grad = safe_interpret(expr.jaxpr, expr.literals, x)
print(f"Actual Grad: {grad[0]}\n"
      f"Interpreted Grad: {estimated_grad[0][0]}\n")
