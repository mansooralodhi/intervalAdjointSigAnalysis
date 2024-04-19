
import jax
from jax import grad
from functools import partial
from src.makeModel.modelRuntime import ModelRuntime

class ModelJaxpr():
    def __init__(self, model_runtime: ModelRuntime):
        self.model_runtime = model_runtime

    def grad_jaxpr_wrt_inputs(self, x = None):
        if x is None:
            x = self.model_runtime.sampleX
        # fixme: if we don't make model_params static then jaxpr need inputs against each nn layer.
        partial_loss_func = partial(self.model_runtime.loss, model_params=self.model_runtime.model_params)
        grad_fn = grad(partial_loss_func)
        expr = jax.make_jaxpr(grad_fn)(x=x)
        return expr

    def grad_jaxpr_wrt_params(self, params=None):
        if params is None:
            params = self.model_runtime.model_params
        partial_loss_func = partial(self.model_runtime.loss, x=self.model_runtime.sampleX)
        grad_fn = grad(partial_loss_func)
        expr = jax.make_jaxpr(grad_fn)(model_params=params)
        return expr

    def forward_jaxpr_wrt_inputs(self, x=None):
        if x is None:
            x = self.model_runtime.sampleX
        # fixme: if we don't make model_params static then jaxpr need inputs against each nn layer.
        partial_loss_func = partial(self.model_runtime.loss, model_params=self.model_runtime.model_params)
        expr = jax.make_jaxpr(partial_loss_func)(x=x)
        return expr

    def forward_jaxpr_wrt_params(self, x=None):
        if x is None:
            x = self.model_runtime.sampleX
        partial_loss_func = partial(self.model_runtime.loss, x=x)
        expr = jax.make_jaxpr(partial_loss_func)(model_params=self.model_runtime.model_params)
        return expr


if __name__ == "__main__":
    runtime = ModelRuntime()
    expr = ModelJaxpr(runtime).forward_jaxpr_wrt_inputs()
    print(expr)
