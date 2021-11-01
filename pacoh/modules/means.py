from abc import abstractmethod

import haiku as hk
from jax import numpy as jnp


class JAXMean(hk.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, x):
        pass


class JAXConstantMean(JAXMean):
    """Mean module with a learnable mean. """
    def __init__(self, initial_constant=0.0):
        super().__init__()
        self.init_constant = initial_constant

    def __call__(self, x):
        # works for both batch or unbatched
        mean = hk.get_parameter("mu", shape=[], dtype=jnp.float32,
                                init=hk.initializers.Constant(self.init_constant))
        return jnp.ones(x.shape) * mean


class JAXZeroMean(JAXMean):
    """Always zero, not learnable. """
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return jnp.zeros(x.shape)