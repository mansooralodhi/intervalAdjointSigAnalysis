
import jax
from jax import grad
from functools import partial
from src.makeModel.runtime import Runtime

class ModelJaxpr(Runtime):
    def __init__(self):
        super(ModelJaxpr, self).__init__()

    def grad_jaxpr_wrt_inputs(self, x = None):
        if x is None:
            x = self.sampleX
        # fixme: if we don't make model_params static then jaxpr need inputs against each nn layer.
        partial_loss_func = partial(self.loss, model_params=self.model_params)
        grad_fn = grad(partial_loss_func)
        expr = jax.make_jaxpr(grad_fn)(x)
        return expr

    def grad_jaxpr_wrt_params(self, params=None):
        if params is None:
            params = self.model_params
        # fixme: this method not working
        partial_loss_func = partial(self.loss, x=self.sampleX)
        grad_fn = grad(partial_loss_func)
        expr = jax.make_jaxpr(grad_fn)(params)
        return expr

    def forward_jaxpr_wrt_inputs(self, x=None):
        if x is None:
            x = self.sampleX
        # fixme: if we don't make model_params static then jaxpr need inputs against each nn layer.
        partial_loss_func = partial(self.loss, model_params=self.model_params)
        expr = jax.make_jaxpr(partial_loss_func)(x)
        return expr

    def forward_jaxpr_wrt_params(self, params=None):
        if params is None:
            params = self.model_params
        # fixme: this method not working
        partial_loss_func = partial(self.loss, x=self.sampleX)
        expr = jax.make_jaxpr(partial_loss_func)(params)
        return expr


if __name__ == "__main__":
    expr = ModelJaxpr().forward_jaxpr_wrt_inputs()
    print(expr)
