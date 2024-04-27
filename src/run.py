


from src.makeModel.modelRuntime import ModelRuntime
from src.makeModel.modelJaxpr import ModelJaxpr
from src.makeModel.modelGrads import ModelGrad
from src.customInterpreter.interpret import safe_interpret
from src.utils import merge_args

modelRuntime = ModelRuntime()
modelJaxpr = ModelJaxpr(modelRuntime)
modelGrads = ModelGrad(modelRuntime)

x = modelRuntime.sampleX
params = modelRuntime.model_params

featuresInterval = modelRuntime.feature_intervals
featureIval = (featuresInterval[0], featuresInterval[1])

loss = modelRuntime.loss(x, params)
expr = modelJaxpr.primal_jaxpr(x, params)
inp = merge_args(x, params)
x_est_loss = safe_interpret(expr.jaxpr, expr.literals, inp)[0]


adjoints = modelGrads.grad(x, params, wrt_arg=(0,1))
expr = modelJaxpr.adjoint_jaxpr(x, params, wrt_arg=(0, 1))
inp = merge_args(x, params)
est_adjoint = safe_interpret(expr.jaxpr, expr.literals, inp)

