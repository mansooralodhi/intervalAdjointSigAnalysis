from flax.training import checkpoints
from pathlib import Path

cwd = Path.cwd()
model_file = cwd.as_posix() + '/checkpoints'


loaded_model_state = checkpoints.restore_checkpoint(ckpt_dir=model_file, target=None)
print(loaded_model_state)


