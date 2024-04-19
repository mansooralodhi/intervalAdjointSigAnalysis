from jax import grad
from functools import partial
from src.makeModel.modelRuntime import ModelRuntime

class ModelGrad(object):
    def __init__(self, model_runtime: ModelRuntime):
        self.model_runtime = model_runtime

    def grad_wrt_inputs(self, x=None):
        if x is None:
            x = self.model_runtime.sampleX
        # fixme: if we don't make model_params static then jaxpr need inputs against each nn layer.
        partial_loss_func = partial(self.model_runtime.loss, model_params=self.model_runtime.model_params)
        return grad(partial_loss_func)(x)

    def grad_wrt_params(self, params=None):
        if params is None:
            params = self.model_runtime.model_params
        # fixme: this method not working
        partial_loss_func = partial(self.model_runtime.loss, x=self.model_runtime.sampleX)
        return grad(partial_loss_func)(params)



if __name__ == '__main__':
    runtime = ModelRuntime()
    grad = ModelGrad(runtime).grad_wrt_inputs()
    print(grad.shape)