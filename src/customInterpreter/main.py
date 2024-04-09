


from src.makeModel.runtime import Runtime
from src.makeModel.jaxpr import ModelJaxpr
from src.makeModel.grads import ModelGrad
from src.customInterpreter.forward_mode import safe_forward_interpret


modelRuntime = Runtime()
modelJaxpr = ModelJaxpr()
modelGrads = ModelGrad()

x = modelRuntime.sampleX
params = modelRuntime.model_params


###################################### Check forward-pass #####################
loss = modelRuntime.loss(x, params)
expr = modelJaxpr.forward_jaxpr_wrt_inputs(x)
estimated_loss = safe_forward_interpret(expr.jaxpr, expr.literals, x)
print(f"Actual Loss: {loss}\n"
      f"Interpreted Loss: {estimated_loss[0]}\n")

##################################### Check reverse-pass #####################

grad = modelGrads.grad_wrt_inputs()
expr = modelJaxpr.grad_jaxpr_wrt_inputs()
estimated_grad = safe_forward_interpret(expr.jaxpr, expr.literals, x)

print(f"Actual Grad: {grad[0]}\n"
      f"Interpreted Grad: {estimated_grad[0][0]}\n")
