


# todo: ...
import warnings
import jax
from torchvision.datasets import MNIST
from src.model.dataset import Dataset
from src.model.runtime import ModelRuntime
from src.interpreter.interpret_v0 import safe_interpret

from tqdm import tqdm

class IntervalOpsValidator(object):
    def __init__(self, model_file: str = ''):
        trainData = MNIST(root='../model/data/mnist', download=True, train=True)
        testData = MNIST(root='../model/data/mnist', download=True, train=False)

        self.trainDataset = Dataset(trainData)
        self.testDataset = Dataset(testData)

        self.modelRuntime = ModelRuntime(model_file) # loads model checkpoints file


    def validate(self):
        ival_adjoints = self._get_ival_adjoints()

        for i in tqdm(range(len(self.testDataset))):
            x, _ = self.testDataset[i]
            sample_adjoints = self.modelRuntime.grad(x.numpy(), self.modelRuntime.model_params, wrt_arg=(0, 1))
            flatParams, _ = jax.tree_flatten(sample_adjoints[1])
            flatParams.insert(0, sample_adjoints[0])
            for j in range(len(flatParams)):
                if not isinstance(ival_adjoints[j], tuple):
                    is_bounded = jax.numpy.all((flatParams[j] - ival_adjoints[j]) == 0)
                else:
                    d1 = flatParams[j].flatten() - ival_adjoints[j][0].flatten()
                    d2 = flatParams[j].flatten() - ival_adjoints[j][1].flatten()
                    is_bounded = jax.numpy.all(jax.numpy.dot(d1, d2) < 0)
                if not is_bounded:
                    raise Exception(f"Sample {i}, layer {j} is out of bound !")
        return True


    def _get_ival_adjoints(self):
        # make jaxpr
        expr = self.modelRuntime.adjoint_jaxpr(self.modelRuntime.sampleX, self.modelRuntime.model_params, wrt_arg=(0, 1))
        # flatten params
        flatParams, _ = jax.tree_flatten(self.modelRuntime.model_params)
        # get feature intervals
        featuresInterval = self.modelRuntime.feature_intervals
        featureIval = (featuresInterval[1], featuresInterval[0])
        # add feature interval with params
        flatParams.insert(0, featureIval)
        # interpret jax expr with interval input and scalar params
        ival_adjoints = safe_interpret(expr.jaxpr, expr.literals, flatParams)
        return ival_adjoints



if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    IntervalOpsValidator().validate()