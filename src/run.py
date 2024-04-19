


from src.makeModel.modelRuntime import ModelRuntime
from src.makeModel.modelJaxpr import ModelJaxpr
from src.makeModel.modelGrads import ModelGrad
from src.customInterpreter.interpret import safe_interpret


modelRuntime = ModelRuntime()
modelJaxpr = ModelJaxpr(modelRuntime)
modelGrads = ModelGrad(modelRuntime)

x = modelRuntime.sampleX
params = modelRuntime.model_params
featuresInterval = modelRuntime.feature_intervals
featuresLowerBound = featuresInterval[0]
featuresUpperBound = featuresInterval[1]
featureIval = (featuresLowerBound, featuresUpperBound)

###################################### Check forward-pass #####################

loss = modelRuntime.loss(x, params)
expr = modelJaxpr.forward_jaxpr_wrt_params(x)
x_ival = featureIval
estimated_loss_x = safe_interpret(expr.jaxpr, expr.literals, params)
# estimated_loss_x_ival = safe_interpret(expr.jaxpr, expr.literals, x_ival)
print(f"Actual Loss: {loss}\n"
      f"Interpreted Loss: {estimated_loss_x[0]}\n")
      # f"Interpreted Loss: {estimated_loss_x_ival[0]}\n")

##################################### Check reverse-pass #####################

# grad = modelGrads.grad_wrt_inputs()
# expr = modelJaxpr.grad_jaxpr_wrt_inputs(params)
# x_ival = featureIval
# estimated_grad_float = safe_interpret(expr.jaxpr, expr.literals, x)
# estimated_grad_interval = safe_interpret(expr.jaxpr, expr.literals, x_ival)
# print(f"Actual Grad: {grad.shape}\n"
#       f"Interpreted Grad LB: {estimated_grad[0][0].shape}\n"
#       f"Interpreted Grad UB: {estimated_grad[0][1].shape}\n")
