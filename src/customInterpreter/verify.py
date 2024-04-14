from src.makeModel.runtime import Runtime
from src.makeModel.grads import ModelGrad
from src.makeModel.jaxpr import ModelJaxpr
from src.customInterpreter.interpret import interpret

################### So far so good ! #############

modelRuntime = Runtime()
modelJaxpr = ModelJaxpr()
modelGrads = ModelGrad()
print("\n\n")

x = modelRuntime.sampleX
params = modelRuntime.model_params

loss = modelRuntime.loss(x, params)
expr = modelJaxpr.forward_jaxpr_wrt_inputs(x)
estimated_loss = interpret(expr.jaxpr, expr.literals, x)

grad = modelGrads.grad_wrt_inputs(x)
expr = modelJaxpr.grad_jaxpr_wrt_inputs(x)
estimated_grad = interpret(expr.jaxpr, expr.literals, x)

print(f"Actual Loss: {loss}\n"
      f"Interpreted Loss: {estimated_loss[0]}\n"
      f"-------------------------------------\n"
      f"Actual Grad: {grad[10]}\n"
      f"Interpreted Grad: {estimated_grad[0][10]}")

