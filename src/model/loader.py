

import os
from flax.training import checkpoints
from src.model.neuralNet import NeuralNet


class ModelLoader(object):
    def __init__(self, model_file: str = 'checkpoints/ckpt_3.0/checkpoint'):
        # fixme: create setup file and add the model file in its packages.
        if model_file == '':
            model_file = 'checkpoints/ckpt_3.0/checkpoint'
            model_file = os.path.join(os.path.dirname(__file__), *model_file.split('/'))
        self.model_file = model_file
        self.model = None
        self.model_params = None
        self.sampleX = None
        self.sampleY = None
        self.feature_intervals = None
        self.load_model_params()

    def define_model(self):
        self.model = NeuralNet()
        self.model = self.model.apply  # model = jax.jit(model.apply)

    def load_model_params(self):
        if self.model is None:
            self.define_model()
        print(f"Loading model file {self.model_file}")
        restored_dict = checkpoints.restore_checkpoint(ckpt_dir=self.model_file, target=None)
        self.sampleX = restored_dict.get("x")
        self.sampleY = restored_dict.get("y")
        self.model_params = restored_dict.get("model_params")
        self.feature_intervals = restored_dict.get("feature_intervals")
        print("Model Loaded.")


if __name__ == "__main__":
    loader = ModelLoader()
    loader.load_model_params()
    print(loader.sampleX.shape)
    print(loader.feature_intervals.shape)