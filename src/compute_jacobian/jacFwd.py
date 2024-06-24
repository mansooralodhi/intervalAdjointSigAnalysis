


from src.model.standaloneNN import inputs, parameters, loss
from src.site_packages.custom_interpreter.api import scalar_interpret, interval_interpret


scalar_x, ival_x = inputs()
params = parameters()
y = loss(scalar_x, params)

print(y)
print("-"*50)

y = scalar_interpret(loss)(scalar_x, params)
y = interval_interpret(loss)(ival_x, scalar_x, params)

