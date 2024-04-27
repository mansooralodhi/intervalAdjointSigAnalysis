import jax
from typing import Union
from src.makeModel.modelRuntime import ModelRuntime

class ModelGrad(object):

    def __init__(self, model_runtime: ModelRuntime):
        self.model_runtime = model_runtime

    def grad(self, *args, wrt_arg: Union[int, tuple]):
        return jax.grad(self.model_runtime.loss, argnums=wrt_arg)(*args)



if __name__ == '__main__':
    runtime = ModelRuntime()
    x = runtime.sampleX
    params = runtime.model_params
    modelgrad = ModelGrad(runtime)
    z = modelgrad.grad(x, params, wrt_arg=(0,1))
    print(x)