from flax import linen

"""
Dense(dim=n)(input=x):
    Input:
        x = (..., p)
    Params:
        w = (p, n)
    return: 
        x @ w = (..., n) 
"""


class NeuralNet(linen.Module):
    n_targets = 10

    @linen.compact
    def __call__(self, x):
        x = linen.Dense(128)(x)
        x = linen.relu(x)
        x = linen.Dense(self.n_targets)(x)
        return x
