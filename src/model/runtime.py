

"""
NB: the only place where we have params is the model itself, hence,
    we can alter the loss function without hesitation.
    thus, runtime loss function can be different form training loss function
    secondly, we can make loss_fn or/and model.appy as jit function, though
    both gives almost same jaxpr.
    lastly, if you want to decorate loss function with jit then define a new 
    loss function outside class and class it from inside class.
    
                    !!! Very Important !!!
    If need to find gradient w.r.t model_params then we need to pass argument 
    for each nn layer. 
    Thus, it might be challenging to find the adjoints of intermediate/internal
    nodes/outputs of the computational graph. 
    For time being, we are making the model_params as static and the input to
    model as variable in order to generate jaxpr with single input instance,
    rather then also passing arguments for each layers/parameter.
    We might need to store the intermediate input to each layer during forward-pass
    and use them during the reverse pass as argument to loss function in order to 
    find the adjoint of each layer and node in reverse-mode. 
"""

import jax
from typing import Union
from src.model.loader import ModelLoader

class ModelRuntime(ModelLoader):
    def __init__(self, model_file: str = ''):
        super(ModelRuntime, self).__init__(model_file)

    def predict(self, x, model_params):
        scores = self.model(model_params, x)
        return jax.numpy.argmax(scores, -1)

    def loss(self, x, model_params):
        # todo: go back to classification problem and refractor the code accoudingly.
        scores = self.model(model_params, x)
        return scores.mean()

    def primal_jaxpr(self, *args):
        return jax.make_jaxpr(self.loss)(*args)

    def adjoint_jaxpr(self, *args, wrt_arg: Union[int, tuple]):
        grad_fn = jax.grad(self.loss, argnums=wrt_arg)
        expr = jax.make_jaxpr(grad_fn)(*args)
        return expr

    def grad(self, *args, wrt_arg: Union[int, tuple]):
        return jax.grad(self.loss, argnums=wrt_arg)(*args)

