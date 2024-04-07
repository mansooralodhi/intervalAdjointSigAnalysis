from flax.training import checkpoints
from pathlib import Path
from src.makeModel.neuralNet import NeuralNet
import jax
from jax import grad

"""
NB: the only place where we have params is the model itself, hence,
    we can alter the loss function without hesitation.
    thus, runtime loss function can be different form training loss function
    secondly, we can make loss_fn or/and model.appy as jit function, though
    both gives almost same jaxpr.
    lastly, if you want to decorate loss function with jit then define a new 
    loss function outside class and class it from inside class.
"""


class Runtime(object):
    def __init__(self, model_file='/checkpoints/ckpt_1.0/checkpoint'):
        self.model_file = Path.cwd().as_posix() + model_file
        self.model = None
        self.model_params = None
        self.sampleX = None
        self.sampleY = None

    def loss(self, x=None, y=None, model_params=None):
        if x is None:
            x = self.sampleX
        scores = self.model(model_params, x)
        return scores.mean()

    def define_model(self):
        self.model = NeuralNet()
        self.model = self.model.apply  # model = jax.jit(model.apply)

    def load_model_params(self):
        if self.model is None:
            self.define_model()
        restored_dict = checkpoints.restore_checkpoint(ckpt_dir=self.model_file, target=None)
        self.model_params = restored_dict.get("model_params")
        self.sampleX = restored_dict.get("x")
        self.sampleY = restored_dict.get("y")

    def get_jaxpr(self):
        if self.model_params is None:
            self.load_model_params()
        grad_fn = grad(self.loss, argnums=2)
        expr = jax.make_jaxpr(grad_fn)(self.sampleX, self.sampleY, self.model_params)
        return expr


if __name__ == "__main__":
    runtime = Runtime()
    print(runtime.get_jaxpr())