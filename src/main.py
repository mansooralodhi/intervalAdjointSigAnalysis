


from src.makeModel.runtime import Runtime
from src.makeModel.jaxpr import ModelJaxpr
from src.makeModel.grads import ModelGrad
from src.customInterpreter.interpret import safe_interpret


modelRuntime = Runtime()
modelJaxpr = ModelJaxpr()
modelGrads = ModelGrad()

x = modelRuntime.sampleX
params = modelRuntime.model_params
featuresInterval = modelRuntime.feature_intervals
featuresLowerBound = featuresInterval[0]
featuresUpperBound = featuresInterval[1]
featureIval = (featuresLowerBound, featuresUpperBound)

###################################### Check forward-pass #####################

loss = modelRuntime.loss(x, params)
expr = modelJaxpr.forward_jaxpr_wrt_inputs(x)
x_ival = featureIval
estimated_loss = safe_interpret(expr.jaxpr, expr.literals, x_ival)
print(f"Actual Loss: {loss}\n"
      f"Interpreted Loss: {estimated_loss[0]}\n")

##################################### Check reverse-pass #####################

grad = modelGrads.grad_wrt_inputs()
expr = modelJaxpr.grad_jaxpr_wrt_inputs()
x_ival = featureIval
estimated_grad = safe_interpret(expr.jaxpr, expr.literals, x_ival)
print(f"Actual Grad: {grad.shape}\n"
      f"Interpreted Grad LB: {estimated_grad[0][0].shape}\n"
      f"Interpreted Grad UB: {estimated_grad[0][1].shape}\n")
