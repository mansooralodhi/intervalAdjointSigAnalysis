

from src.makeModel.dataset import Dataset
from src.makeModel.dataloader import Dataloader
from src.makeModel.neuralNet import NeuralNet
from src.makeModel.trainer import train_model

import jax
import optax
from pathlib import Path
from flax.training import train_state
from flax.training import checkpoints
from torchvision.datasets import MNIST


trainData = MNIST(root='data/mnist', download=True, train=True)
testData = MNIST(root='data/mnist', download=True, train=False)

batch_size = 128
trainDataset = Dataset(trainData)
trainDataloader = Dataloader(trainDataset, batch_size=batch_size)
testDataset = Dataset(testData)
testDataloader = Dataloader(testDataset, batch_size=batch_size)

learning_rate = 0.01
key = jax.random.key(0)
x, y = trainDataset[0][0], trainDataset[0][1]
model = NeuralNet()
model_params_dict = model.init(key, x)
model_state = train_state.TrainState.create(apply_fn=model.apply,
                                            params=model_params_dict,
                                            tx=optax.adam(learning_rate=learning_rate))

epochs = 1
train_output_dict = train_model(model_state, trainDataloader, testDataloader, num_epochs=epochs)
trained_model = train_output_dict.get('model_state')

ckpt = dict(model_params=trained_model.params, x=jax.numpy.asarray(x), y=jax.numpy.asarray(y),
            feature_intervals=trainDataset.feature_intervals())


# fixme: put this checkpoint code in Trainer class
model_file = Path.cwd().as_posix() + '/checkpoints'
checkpoints.save_checkpoint(
    ckpt_dir=model_file,  # Folder to save checkpoint in
    target=ckpt,  # What to save. To only save parameters, use model_state.params
    step=0,  # Training step or other metric to save best model on
    prefix="ckpt_3.",  # Checkpoint file name prefix
    overwrite=True,  # Overwrite existing checkpoint files
)