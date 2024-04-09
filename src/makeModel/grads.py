from jax import grad
from functools import partial
from src.makeModel.runtime import Runtime

class ModelGrad(Runtime):
    def __init__(self):
        super(ModelGrad, self).__init__()

    def grad_wrt_inputs(self, x=None):
        if x is None:
            x = self.sampleX
        # fixme: if we don't make model_params static then jaxpr need inputs against each nn layer.
        partial_loss_func = partial(self.loss, model_params=self.model_params)
        return grad(partial_loss_func)(x)

    def grad_wrt_params(self, params=None):
        if params is None:
            params = self.model_params
        # fixme: this method not working
        partial_loss_func = partial(self.loss, x=self.sampleX)
        return grad(partial_loss_func)(params)



if __name__ == '__main__':
    grad = ModelGrad().grad_wrt_inputs()
    print(grad.shape)