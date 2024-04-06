
import optax
from collections import defaultdict


def adam_optimzer(learrning_rate: float):
    return optax.adam(learning_rate=learrning_rate)

def update_params(learning_rate: float, params_dict: dict[str], gradient: dict[str]) -> dict[str]:
    params = params_dict.get('params')
    gradient = gradient.get('params')
    updated_params = defaultdict(lambda: defaultdict(dict))
    for layer in params:
        for para in params[layer]:
            updated_params['params'][layer][para] = params[layer][para] - learning_rate * gradient[layer][para]
    return updated_params


