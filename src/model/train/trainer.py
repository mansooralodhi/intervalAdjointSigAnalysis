import jax
import optax
import torch
from tqdm import tqdm
import jax.numpy as jnp
from flax.training.train_state import TrainState

@jax.jit
def loss_accu_fn(x: torch.Tensor, y: torch.Tensor, params_dict: dict[str], model_state: TrainState) -> tuple[jax.Array, jax.Array]:
    """
    NB: we need params_dict separately in order to find gradient w.r.t argument.
    Args:
        x: (b, p) | (b, ...)
        y: (b,)
        params_dict: {'params': ...}
        model_state
    Return:
        total_loss: scalar
        total_accu: scalar
    """
    scores = model_state.apply_fn(params_dict, x)                # (b, p)
    y_onehot = jax.nn.one_hot(y, scores.shape[-1])               # (b, p)
    loss = optax.sigmoid_binary_cross_entropy(scores, y_onehot)  # (b,)
    loss = loss.mean()                                           # ()

    y_predicted = jnp.argmax(scores, -1)                         # (b,)
    accu = jnp.mean(y_predicted == y)                            # ()
    return loss, accu

@jax.jit
def train_step(x: torch.Tensor, y: torch.Tensor, model_state: TrainState) -> tuple[TrainState, jax.Array, jax.Array]:
    """
    Args:
        x: (b, p) | (b, ...)
        y: (b,)
        model_state
    Return:
        model_state
        total_loss: scalar jax array
        total_accu: scalar jax array
    """
    (loss, acc), grads = jax.value_and_grad(loss_accu_fn, argnums=2, has_aux=True)(x, y, model_state.params, model_state)
    model_state = model_state.apply_gradients(grads=grads)
    return model_state, loss, acc

@jax.jit
def eval_step(x: torch.Tensor, y:torch.Tensor, model_state: TrainState) -> tuple[jax.Array, jax.Array]:
    """
    Args:
        x: (b, p) | (b, ...)
        y: (b,)
        model_state
    Return:
        total_loss: scalar jax array
        total_accu: scalar jax array
    """
    loss, acc = loss_accu_fn(x, y, model_state.params, model_state)
    return loss, acc


def train_model(model_state: TrainState, trainDataloader, testDataloader, num_epochs: int) -> dict[str]:

    """
    Args:
        model_state
        trainDataloader: torch.utils.data.Dataloader
        testDataloader: torch.utils.data.Dataloader
        num_epochs: int
    Return:
        dict:
            state_model
            trainAccuray: list
            trainLoss:    list
            testAccuray:  list
            testLoss:     list
    """

    # todo:
    #   (1) make Trainer class and add train_model in it
    #   (2) save the loss and accuracy with model checkpoint
    #   (3) save multiple checkpoints with best accuracies
    #   (4) add tensorboard feature to monitor loss and accuracy,

    training_accuracy = []
    training_loss = []

    testing_loss = []
    testing_accuracy = []

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        val_batch_loss, val_batch_accuracy = 0, 0
        train_batch_loss, train_batch_accuracy = 0, 0

        for x, y in trainDataloader:
            x, y = jnp.asarray(x), jnp.asarray(y)
            model_state, loss, acc = train_step(x, y, model_state)
            train_batch_loss += loss
            train_batch_accuracy += acc

        for x, y in testDataloader:
            x, y = jnp.asarray(x), jnp.asarray(y)
            val_loss, val_acc = eval_step(x, y, model_state)
            val_batch_loss += val_loss
            val_batch_accuracy += val_acc

        # Loss for the current epoch
        epoch_train_loss = train_batch_loss / len(trainDataloader)
        epoch_val_loss = val_batch_loss / len(testDataloader)

        # Accuracy for the current epoch
        epoch_train_acc = train_batch_accuracy / len(trainDataloader)
        epoch_val_acc = val_batch_accuracy / len(testDataloader)

        testing_loss.append(epoch_val_loss)
        testing_accuracy.append(epoch_val_acc)

        training_loss.append(epoch_train_loss)
        training_accuracy.append(epoch_train_acc)

        print(f"Epoch: {epoch + 1}, loss: {epoch_train_loss:.2f}, acc: {epoch_train_acc:.2f} val loss: {epoch_val_loss:.2f} val acc {epoch_val_acc:.2f} ")

    return dict(model_state=model_state, trainAccuracy=training_accuracy, trainLoss=training_loss,
                testAccuracy=testing_accuracy, testLoss=testing_loss)


