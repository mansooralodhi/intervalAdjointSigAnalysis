from flax.training import checkpoints
import jax
from jax import grad
from functools import partial

from src.makeModel.neuralNet import NeuralNet

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


class Runtime(object):
    # fixme: use relative path and find the relative file.
    def __init__(self, model_file="D:/TGGS/Stce/repositories/intervalAdjointSigAnalysis/src/makeModel/checkpoints"
                                  "/ckpt_3.0/checkpoint"):
        self.model_file = model_file
        self.model = None
        self.model_params = None
        self.sampleX = None
        self.sampleY = None
        self.feature_intervals = None
        self.load_model_params()

    def predict(self, x):
        scores = self.model(self.model_params, x)
        return jax.numpy.argmax(scores, -1)

    def loss(self, x=None, model_params=None):
        scores = self.model(model_params, x)
        return scores.mean()

    def define_model(self):
        self.model = NeuralNet()
        self.model = self.model.apply  # model = jax.jit(model.apply)

    def load_model_params(self):
        self.define_model()
        restored_dict = checkpoints.restore_checkpoint(ckpt_dir=self.model_file, target=None)
        self.model_params = restored_dict.get("model_params")
        self.sampleX = restored_dict.get("x")
        self.sampleY = restored_dict.get("y")
        self.feature_intervals = restored_dict.get("feature_intervals")
        print("Model Loaded.")





if __name__ == "__main__":
    runtime = Runtime()
    runtime.load_model_params()
    loss = runtime.loss(runtime.sampleX, runtime.model_params)
    print(loss)
    print(runtime.sampleX.shape)
    print(runtime.feature_intervals.shape)