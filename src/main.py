

from src.dataset import Dataset
from src.dataloader import Dataloader
from src.neuralNet import NeuralNet
from src.optimizer import adam_optimzer
from src.trainer import train_model

import jax
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
model = NeuralNet()
inpSample = trainDataset[0][0]
key = jax.random.key(0)
model_params_dict = model.init(key, inpSample)
print(jax.make_jaxpr())
model_state = train_state.TrainState.create(apply_fn=model.apply, params=model_params_dict, tx=adam_optimzer(learning_rate))

epochs=5
train_output_dict = train_model(model_state, trainDataloader, testDataloader, num_epochs=epochs)
trained_model = train_output_dict.get('model_state')

cwd = Path.cwd()
model_file = cwd.as_posix() + '/checkpoints'
checkpoints.save_checkpoint(
    ckpt_dir=model_file,  # Folder to save checkpoint in
    target=trained_model,  # What to save. To only save parameters, use model_state.params
    step=1,  # Training step or other metric to save best model on
    prefix="my_model",  # Checkpoint file name prefix
    overwrite=True,  # Overwrite existing checkpoint files
)