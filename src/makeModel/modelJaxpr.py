
import jax
from typing import Union
from src.makeModel.modelRuntime import ModelRuntime

class ModelJaxpr():

    def __init__(self, model_runtime: ModelRuntime):
        self.model_runtime = model_runtime

    def primal_jaxpr(self, *args):
        return jax.make_jaxpr(self.model_runtime.loss)(*args)

    def adjoint_jaxpr(self, *args, wrt_arg: Union[int, tuple]):
        grad_fn = jax.grad(self.model_runtime.loss, argnums=wrt_arg)
        expr = jax.make_jaxpr(grad_fn)(*args)
        return expr


if __name__ == "__main__":
    runtime = ModelRuntime()
    x = runtime.sampleX
    params = runtime.model_params
    expr = ModelJaxpr(runtime).adjoint_jaxpr(x, params, wrt_arg=(0,1))
    print(expr)
