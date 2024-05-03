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
    """
    using nn.linen.compact would set default layer/kernel names
    else the name of layer would be set according to the name of
    the variable name using in setup method.
    """
    n_targets = 10
    """
    Problems: There appear to be a problem with the usage of relu activation function.
               With relu, we need to switch input intervals in order to get correct interval output.
    """
    @linen.compact
    def __call__(self, x):
        x = linen.Dense(128)(x)
        x = linen.tanh(x)
        x = linen.Dense(self.n_targets)(x)
        return x

