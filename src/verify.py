from src.makeModel.modelRuntime import ModelRuntime
from src.makeModel.modelGrads import ModelGrad
from src.makeModel.modelJaxpr import ModelJaxpr
from src.customInterpreter.interpret import interpret

################### So far so good ! #############

modelRuntime = ModelRuntime()
modelJaxpr = ModelJaxpr(modelRuntime)
modelGrads = ModelGrad(modelRuntime)
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

